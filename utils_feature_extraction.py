from pyspark.ml.feature import VectorAssembler
import pyspark.ml.feature
from pyspark.ml.functions import vector_to_array
import pyspark.sql.functions as F
import pyspark.sql.types as T
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
import numpy as np


def filter_by_correlation(corr_pdf_target, corr_lim, target):
    log_idx = [False] + list(abs(corr_pdf_target[target]) >= corr_lim)
    cols_filt = list(corr_pdf_target.columns[log_idx])
    cols_filt.remove(target)
    return cols_filt


# Group features from laboratory data and field data by correlation to SOH and collinearity
def group_by_correlation(corr_pdf_features, corr_pdf_target, corr_lim, colin_lim, target):
    corr_pdf_target = corr_pdf_target.sort_values(by=[target], ascending=True)

    cols_mutual = list(set(corr_pdf_target.columns) & set(corr_pdf_features.columns))

    corr_pdf_features = corr_pdf_features[cols_mutual]
    corr_pdf_features = corr_pdf_features.loc[corr_pdf_features.col_names.isin(cols_mutual)]

    corr_pdf_target = corr_pdf_target[cols_mutual]
    corr_pdf_target = corr_pdf_target.loc[corr_pdf_target.col_names.isin(cols_mutual)]

    corr_dict = {}
    skip_var = []

    for _, row_corr in corr_pdf_target.iterrows():
        if row_corr['col_names'] in skip_var:
            continue

        if row_corr['col_names'] == target:
            continue

        if not abs(row_corr[target]) >= corr_lim:
            continue

        corr_dict[row_corr.col_names] = []
        skip_var.append(row_corr.col_names)

        row_corr_features = corr_pdf_features.loc[corr_pdf_features.col_names == row_corr.col_names].iloc[0]
        for col, col_val in row_corr_features.items():
            if col in ['col_names', target] or col == row_corr['col_names']:
                continue

            if col in skip_var:
                continue

            if abs(col_val) > colin_lim:
                corr_dict[row_corr.col_names].append(col)
                skip_var.append(col)
    return corr_dict


# Compress features with PCA method
def compress_with_pca(df, feat_groups, feat_names=tuple(), n_pca=1):
    df_pca = df
    cols_pca = []

    for c, features in enumerate(feat_groups):
        # assemble feature columns in vector column
        vector_col = 'pca_features'
        assembler = VectorAssembler(inputCols=features, outputCol=vector_col, handleInvalid='skip')
        df_pca = assembler.transform(df_pca)

        # apply PCA on vector column
        if feat_names:
            col_pca = feat_names[c]
        else:
            col_pca = f'pca{len(features)}'

        cols_pca.append(col_pca)

        model = pyspark.ml.feature.PCA(k=n_pca, inputCol=vector_col, outputCol=col_pca).fit(df_pca)

        df_pca = model.transform(df_pca)
        df_pca = df_pca.withColumn(col_pca, vector_to_array(col_pca)[0])
        df_pca = df_pca.drop(vector_col)

    df_pca = df_pca.select(['readout_id_encrypted'] + cols_pca)
    return df_pca


# Compress features with PCA method
def compress_with_pca_per_vehicle(df, corr_dict, target, n_pca=1):
    df_pca = df
    cols_id = ['readout_id_encrypted', 'van', target]
    df_collect = df.select(cols_id)

    for k, v in corr_dict.items():
        v.append(k)
        features = v

        col_pca = f'pca_n{len(v)}_'+k
        schema_pdf = T.StructType()
        schema_pdf.add('readout_id_encrypted', T.StringType())
        schema_pdf.add('van', T.StringType())
        schema_pdf.add(col_pca, T.FloatType())

        @F.pandas_udf(schema_pdf, F.PandasUDFType.GROUPED_MAP)
        def compute_pca(pdf):
            pca = PCA(n_components=1)
            pdf[col_pca] = pca.fit_transform(pdf[features])
            return pdf[['readout_id_encrypted', 'van', col_pca]]

        df_select = df_pca.select(cols_id + features)
        df_pca_select = df_select.groupBy('van').apply(compute_pca)
        df_pca_select = df_pca_select.drop('van')
        df_collect = df_collect.join(df_pca_select, on='readout_id_encrypted', how='inner')

    return df_collect


# Compress features with PLS method
def compress_with_pls_per_vehicle(df, corr_dict, target, n_pca=1):
    df_pls = df
    cols_id = ['readout_id_encrypted', 'van', target]
    df_collect = df.select(cols_id)

    for k, v in corr_dict.items():
        v.append(k)
        features = v

        col_pls = f'pls_n{len(v)}_' + k
        schema_pdf = T.StructType()
        schema_pdf.add('readout_id_encrypted', T.StringType())
        schema_pdf.add(col_pls, T.FloatType())

        @F.pandas_udf(schema_pdf, F.PandasUDFType.GROUPED_MAP)
        def compute_pls(pdf):
            pls = PLSRegression(n_components=1, tol=1e-09, max_iter=1000, scale=True)
            X = pdf[features].to_numpy()
            y = pdf[target].to_numpy().reshape((-1, 1))
            try:
                pdf[col_pls] = pls.fit_transform(X, y)[0]
            # catch if all features constant/zero
            except ValueError:
                pdf[col_pls] = pdf[k]
            return pdf[['readout_id_encrypted', col_pls]]

        df_select = df_pls.select(cols_id + features)
        df_pls_select = df_select.groupBy('van').apply(compute_pls)
        df_collect = df_collect.join(df_pls_select, on='readout_id_encrypted', how='inner')
    return df_collect