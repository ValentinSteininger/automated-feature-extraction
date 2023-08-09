import os
import re
import pandas as pd  # noqa
import numpy as np
from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession
from myproject.analysis.utils.utils_datasets import Utils
import pyspark.sql.functions as F
import pyspark.sql.types as T


# calculate correlation matrix
def calc_corr(df, cols, method='pearson', filt_nan=False):
    df = df.select(cols)

    vector_col = "corr_features"
    assembler = VectorAssembler(inputCols=df.columns, outputCol=vector_col, handleInvalid='skip')
    df_vector = assembler.transform(df).select(vector_col)

    # get correlation matrix
    matrix = Correlation.corr(df_vector, vector_col, method)
    corr_np = matrix.collect()[0][matrix.columns[0]].toArray()  # noqa

    if filt_nan:
        filt_row = []
        for c, row in enumerate(corr_np):
            row = np.delete(row, c)
            if np.all(np.isnan(row)):
                filt_row.append(c)

        corr_np = np.delete(corr_np, filt_row, axis=0)
        corr_np = np.delete(corr_np, filt_row, axis=1)

        for idx in reversed(filt_row):
            cols.pop(idx)

    corr_pdf = pd.DataFrame(data=corr_np, columns=cols)
    corr_pdf['variable_idx'] = [i for i in range(len(cols))]
    corr_pdf['col_names'] = cols

    spark = SparkSession.builder.appName("pandas to spark").getOrCreate()
    corr_df = spark.createDataFrame(corr_pdf)
    corr_df = corr_df.select(['col_names'] + cols)
    corr_df = corr_df.orderBy('variable_idx', ascending=True)
    corr_df = corr_df.drop('variable_idx')
    return corr_df


def calc_corr_per_vehicle(df, cols, method='pearson', filt_nan=False, df_filtcorr=None):
    def fun_correlation(df):
        schema_pdf = T.StructType()
        cols = df.schema.names
        cols.remove('van')
        for col in cols:
            schema_pdf.add(col, T.FloatType())
        schema_pdf = schema_pdf.add('col_idx', T.IntegerType())
        schema_pdf = schema_pdf.add('van', T.StringType())

        @F.pandas_udf(schema_pdf, F.PandasUDFType.GROUPED_MAP)
        def calc_corr(pdf):
            corr_pdf = pdf.corr(method=method)
            corr_pdf['col_idx'] = [i for i in range(len(corr_pdf.columns))]
            corr_pdf['van'] = pdf['van'][0]
            return corr_pdf

        corr_df = df.groupBy('van').apply(calc_corr)
        return corr_df

    df = df.select(['van'] + cols)

    # compute correlation for each van17
    corr_df_vehicles = fun_correlation(df)

    # mask nan values from other dataframe correlation matrix to compare means for feature extraction
    if df_filtcorr:
        # sort filter cols
        cols_sort = []
        for col1 in corr_df_vehicles.schema.names:
            for col2 in df_filtcorr.schema.names:
                if col2 in col1:
                    df_filtcorr = df_filtcorr.withColumnRenamed(col2, col1)
                    cols_sort.append(col1)
                    break

        df_filtcorr = df_filtcorr.select(cols_sort)
        corr_df_filt = fun_correlation(df_filtcorr)
        cols_mask = corr_df_filt.schema.names
        cols_mask.remove('van')
        cols_mask.remove('col_idx')

        # set all non NaN values to zero
        for col in cols_mask:
            corr_df_filt = corr_df_filt.withColumn(col, F.when(F.col(col).isNotNull(), 0).otherwise(None))
            corr_df_filt = corr_df_filt.withColumnRenamed(col, 'mask_'+col)

        corr_df_vehicles = corr_df_vehicles.join(corr_df_filt, on=['van', 'col_idx'], how='inner')

        # add col to mask col to overwrite with nan values
        for col in cols_mask:
            corr_df_vehicles = corr_df_vehicles.withColumn(col, F.col(col) + F.col('mask_'+col))
            corr_df_vehicles = corr_df_vehicles.drop('mask_'+col)

    # compute mean over all vehicles
    exprs = [F.mean(col).alias(col) for col in cols]
    corr_df = corr_df_vehicles.groupBy('col_idx').agg(*exprs)

    # order corr matrix and add name column
    spark = SparkSession.builder.appName("pandas to spark").getOrCreate()
    colname_df = spark.createDataFrame([(c, col) for c, col in enumerate(cols)], schema=['col_idx', 'col_names'])
    corr_df = corr_df.join(colname_df, on='col_idx')

    corr_df = corr_df.select(cols+['col_names', 'col_idx'])
    corr_df = corr_df.orderBy('col_idx')

    if filt_nan:
        corr_pdf = corr_df.toPandas()
        for c, col in reversed(list(enumerate(corr_pdf.columns[:-2]))):
            if corr_pdf[col].drop(c).isnull().values.all():
                corr_pdf.drop(col, axis=1, inplace=True)
                corr_pdf.drop(c, axis=0, inplace=True)

        corr_df = spark.createDataFrame(corr_pdf)

    cols_sel = corr_df.schema.names
    cols_sel.remove('col_names')

    corr_df = corr_df.select(['col_names']+cols_sel)
    corr_df = corr_df.orderBy('col_idx')
    corr_df = corr_df.drop('col_idx')
    return corr_df
