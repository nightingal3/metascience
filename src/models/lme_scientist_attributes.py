import pandas as pd
import statsmodels.api as sms
import statsmodels.formula.api as smf

if __name__ == "__main__":
    df = pd.read_csv("data/merged_data.csv")
    md_1NN = smf.mixedlm("ll_diff_1NN ~ num_papers + median_collaborators",df, groups=df["field"])
    md_proto = smf.mixedlm("ll_diff_proto ~ num_papers + median_collaborators", df, groups=df["field"])
    mdf_1NN = md_1NN.fit()
    mdf_proto = md_proto.fit()
    print("1NN")
    print(mdf_1NN.summary())
    print(mdf_1NN.aic)
    print(mdf_1NN.bic)
    print("Prototype")
    print(mdf_proto.summary())
    print(mdf_proto.aic)
    print(mdf_proto.bic)