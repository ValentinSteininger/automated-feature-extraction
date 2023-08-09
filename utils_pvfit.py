import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from pyspark.sql.functions import pandas_udf, col, udf
import pyspark.sql.types as T


def fit_columns(df):
    def gen_fun(var_type, *args):
        @pandas_udf(T.ArrayType(T.FloatType()))
        def exec_func(*args):
            pdf = pd.concat(args, axis=1)
            return pdf.apply(calc_pvfit, axis=1, result_type='reduce', args=(var_type,))
        return exec_func(*args)

    df = df.withColumn('pars_dauer_soc', gen_fun(0, *[col(f'dauer_soc_{i}_s') for i in range(1, 11)]))
    df = df.drop(*[f'dauer_soc_{i}_s' for i in range(1, 11)])

    for k in range(1, 7):  # noqa
        df = df.withColumn(f'pars_dauer_soc_temp{k}', gen_fun(0, *[col(f'dauer_soc_{i}_temperatur_{k}_s') for i in range(1, 11)]))

    for k in range(1, 11):  # noqa
        df = df.withColumn(f'pars_dauer_temp_soc{k}', gen_fun(1, *[col(f'dauer_soc_{k}_temperatur_{i}_s') for i in range(1, 7)]))

    df = df.drop(*[f'dauer_soc_{i}_temperatur_{j}_s' for i in range(1, 11) for j in range(1,7)])

    df = df.withColumn('pars_dauer_temp', gen_fun(1, *[col(f'dauer_temperatur_{i}_s') for i in range(1, 7)]))
    df = df.drop(*[f'dauer_temperatur_{i}_s' for i in range(1, 7)])

    df = df.withColumn('pars_zaehler_ladung', gen_fun(2, *[col(f'zaehler_ladungsbereich_{i}_n') for i in range(1, 8)]))
    df = df.drop(*[f'zaehler_ladungsbereich_{i}_n' for i in range(1, 8)])

    df = df.withColumn('pars_zaehler_entladung', gen_fun(2, *[col(f'zaehler_entladungsbereich_{i}_n') for i in range(1, 8)]))
    df = df.drop(*[f'zaehler_entladungsbereich_{i}_n' for i in range(1, 8)])

    df = df.withColumn('pars_ladung_temp', gen_fun(1, *[col(f'ladungszaehler_temperaturbereich_{i}_ah') for i in range(1, 7)]))
    df = df.drop(*[f'ladungszaehler_temperaturbereich_{i}_ah' for i in range(1, 7)])

    df = df.withColumn('pars_entladung_temp', gen_fun(1, *[col(f'entladungszaehler_temperaturbereich_{i}_ah') for i in range(1, 7)]))
    df = df.drop(*[f'entladungszaehler_temperaturbereich_{i}_ah' for i in range(1, 7)])
    return df


def calc_pvfit(pdf_row, f_id):
    width = 0.001
    a = 10
    b = 10
    w = np.hstack((np.linspace(0, width, a), np.linspace(width+0.1, 20 - (width + 0.1), b), np.linspace(20 - width, 20, a)))/20

    bin_ser = pdf_row.to_numpy()
    # add pseudo bins
    bin_ser = np.insert(bin_ser, 0, 0)
    bin_ser = np.append(bin_ser, 0)

    # bin dict
    bin_dict = {
        0: np.array([5, 15, 25, 35, 45, 55, 65, 75, 85, 95]),
        1: np.array([-7.5, 10, 30, 47.5, 62.5, 80]),
        2: np.array([0.55, 1.65, 2.75, 4.4, 6.6, 8.8, 10])
    }

    bin_add_pseudo_dict = {
        0: 10,
        1: 10,
        2: 0.5
    }

    thresh_dict = {
        0: 5,
        1: 5,
        2: 0.5
    }

    bin_pos = bin_dict[f_id]
    bin_pos = np.insert(bin_pos, 0, bin_pos[0] - bin_add_pseudo_dict[f_id])
    bin_pos = np.append(bin_pos, bin_pos[-1] + bin_add_pseudo_dict[f_id])

    pars, errors = run_fit(bin_pos, bin_ser, thresh_dict[f_id], width, a, b)
    return errors + pars


def run_fit(bin_pos, bin_ser, thresh, width, a, b):
    # refinement control
    weighted_refine = np.hstack((np.linspace(0, width, a), np.linspace(width+0.1, 20-(width+0.1),b), np.linspace(20-width, 20, a)))

    pars = np.zeros((len(bin_pos)-2, 4))
    pars[:, 2], pars[:, 3] = 1, 0.5
    pars[:, 1] = bin_pos[1:-1]

    bin_pos_fine, bin_ser_fine = targeted_finesteps(bin_pos, bin_ser, weighted_refine)
    peaks, n = findpeaks_mod(bin_pos_fine, bin_ser_fine)
    in_val, min_lim, max_lim = initial_guess(bin_pos_fine, bin_ser_fine, n, peaks, thresh)

    def find_nearest(array, value):
        idx = (np.abs(array - value)).argmin()
        return idx

    max_iter = 50000
    funcx = fun_generator(n)
    try:
        popt = curve_fit(funcx, bin_pos_fine, bin_ser_fine, p0=in_val, bounds=(min_lim, max_lim), max_nfev=max_iter)
        mse = calc_mse(funcx(bin_pos[1:-1], *popt[0]), bin_ser[1:-1])
        mape = calc_mape(funcx(bin_pos[1:-1], *popt[0]), bin_ser[1:-1])

        indices = list(map(lambda x: find_nearest(bin_pos[1:-1], x), bin_pos_fine[peaks]))
        pars[indices, :] = popt[0].reshape(-1, 4)
    except RuntimeError:
        mse = mape = -1

    return pars.reshape(-1).tolist(), [mse, mape]


def pv_fun(x, amp, mean, sgm, aph):
    return ((1 - aph) * amp / (sgm * 1.772)) * (np.exp(-((x - mean) ** 2) / (2.773 * sgm ** 2))) + (
            aph * amp / 3.141) * (sgm / (((x - mean) ** 2) + ((sgm ** 2))))


def fun_generator(n):
    def fun(x, *args):
        f = pv_fun(x, args[0], args[1], args[2], args[3])
        for i in range(1, n):
            f += pv_fun(x, args[4*i], args[4*i+1], args[4*i+2], args[4*i+3])
        return f
    return fun


def initial_guess(bin_pos, bin_ser, n, peaks, thresh):
    ampl = bin_ser[peaks] # peak heights - amplitude
    mean = bin_pos[peaks]  # positions of bins
    sigma = np.ones(n)
    alpha = np.ones(n)*0.5
    ip_val = np.vstack((ampl, mean, sigma, alpha)).transpose().ravel()
    # creating bounds for components to remain fitted especially near to peaks
    min_lim = []
    max_lim = []
    search_bound_min = ()
    for i in range(n):
        soc_peakloc = bin_pos[peaks[i]]
        min_lim.append(0.0) #amin
        min_lim.append(soc_peakloc-thresh) #bmin
        min_lim.append(1e-2) #cmin
        min_lim.append(0.0) #dmin
        max_lim.append(bin_ser[peaks[i]]*20+1e-2) #amax
        max_lim.append(soc_peakloc+thresh) #bmax
        max_lim.append(100) #cmax
        max_lim.append(1) #dmax
    min_lim = np.asarray(min_lim)
    max_lim = np.asarray(max_lim)
    return ip_val, min_lim, max_lim


def targeted_finesteps(x, y, w):
    x_diff = np.diff(x)
    y_diff = np.diff(y)
    x_mod, y_mod = [],[]
    l = int(max(w))
    w = w[:-2]
    for i in range(len(x)-1):
        for j in w:
            x_mod.append(x[i] + x_diff[i]*j/l)
            y_mod.append(y[i] + y_diff[i]*j/l)
    return np.asarray(x_mod), np.asarray(y_mod)


def findpeaks_mod(x, df):
    tmp = np.round(np.gradient(df/max(abs(df))*10000, x))
    indx = []
    d_data = np.round(np.ediff1d(tmp), 1)
    for i in range(len(d_data)-1):
        if d_data[i] == 0 and d_data[i+1] < 0:
            indx.append(i)
    if len(indx) == 0:
        # use middle value as peak for horizontal
        indx.append(int(.5*len(x)))
    peaks = np.asarray(indx)
    return peaks, len(peaks)


def calc_mse(x, x_real):
    rel = np.square(x_real - x)
    return round(rel.sum()/len(x), 2)


def calc_mape(x, x_real):
    rel = np.divide((x_real- x), x_real, out=np.zeros_like((x_real- x)), where=x_real!=0)
    return np.round(sum(np.abs(rel))/(len(x)-np.count_nonzero(x_real==0)),2)
