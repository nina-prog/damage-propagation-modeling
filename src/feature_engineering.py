def correlation_boundary_drop_list(correlation, boundary):
    """Generate the list of features which fit inside the given correlation boundary.
    :param correlation: correlation matrix
    :type correlation: pd.dataFrame
    :param boundary: correlation boundary to evaluate by
    :type boundary: float

    :return: List of features to be dropped
    :rtype: list[str] or list[index]
    """
    if boundary > 1 or boundary < 0:
        raise ValueError("Correlation boundary must be between 0 and 1")

    corr_rul = correlation["RUL"]
    cd = corr_rul.dropna()
    list_corr = []
    for i in range(0, len(cd)):
        if cd.iloc[i] < boundary and cd.iloc[i] > -boundary:
            list_corr.append(cd.index[i])

    cdd = cd.drop(list_corr, inplace=False)
    return list_corr
