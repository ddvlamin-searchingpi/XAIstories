from scipy.stats import kendalltau


def rank(true_rank_df, extracted_rank_df, rank_by="Shap value"):
    #sort by feature attribution value and and add rank to feature names
    true_rank_df = true_rank_df.copy()
    true_rank_df = true_rank_df.sort_values(by=rank_by, key=lambda s: s.abs(), ascending=False)
    true_rank_df["rank"] = range(len(true_rank_df))

    #sort features according to the extracted rank
    extracted_rank_df = extracted_rank_df.sort_values(by="rank", ascending=True)

    extracted_ranked_features = extracted_rank_df["Feature name"].values
    true_ranked_features = true_rank_df["Feature name"].values

    #only take into account features in common when computing the rank metric
    common_feature_set = set(true_ranked_features) & set(extracted_ranked_features)

    in_feature_set = lambda feature_name: feature_name in common_feature_set
    extracted_ranked_features = list(filter(in_feature_set, extracted_ranked_features))
    true_ranked_features = list(filter(in_feature_set, true_ranked_features))

    true_rank_df = true_rank_df.set_index("Feature name")
    tau, p = kendalltau(
        true_rank_df.loc[extracted_ranked_features]["rank"].values,
        true_rank_df.loc[true_ranked_features]["rank"].values
    )

    return tau, p


