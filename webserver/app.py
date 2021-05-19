from flask_bootstrap import Bootstrap
from flask import (
    Flask,
    render_template,
    Response,
    request,
    url_for,
    redirect,
    flash,
    jsonify,
)
from werkzeug.utils import secure_filename
import os
import time
import sys
from webserver_utils import (
    UPLOAD_FOLDER,
    ensure_folder,
    get_current_time,
    random_string,
    generate_report_dict_list,
)
import pandas as pd
import pickle
from mrp7pred.mrp7pred import MRP7Pred


app = Flask(__name__)

bootstrap = Bootstrap(app)
current_data = ""

# @app.route("/home", methods=["GET", "POST"])
@app.route("/", methods=["GET", "POST"])
def home():
    return render_template("base.html")


def run_pred(df, *clf_dir):
    if len(clf_dir) != 2:
        raise ValueError("Must provide both substrate and modulator models!")
    m7p_mod = MRP7Pred(clf_dir=clf_dir[0])
    out_mod = m7p_mod.predict(
        compound_df=df,
        prefix=f"{get_current_time()}",
    )
    df_feats = m7p_mod.featurized_df
    m7p_sub = MRP7Pred(clf_dir=clf_dir[1])
    out_sub = m7p_sub.predict(
        compound_df=df,
        prefix=f"{get_current_time()}",
        featurized_df=df_feats,
    )

    out_mod["substrate_score"] = out_sub["score"]
    out_mod = out_mod.rename(columns={"score": "modulator_score"})
    out = out_mod.sort_values(by=["modulator_score"], ascending=False)
    report_d_l = generate_report_dict_list(out)
    return report_d_l


@app.route("/wait", methods=["GET", "POST"])
def wait():
    return render_template("wait.html")


@app.route("/run", methods=["GET", "POST"])
def run():
    if request.method == "POST":
        ensure_folder(UPLOAD_FOLDER)
        ts = get_current_time()
        rs = random_string(10)
        random_folder = f"{ts}_{rs}"
        ensure_folder(f"{UPLOAD_FOLDER}/{random_folder}")
        app.config["UPLOAD_FOLDER"] = f"{UPLOAD_FOLDER}/{random_folder}"
        file = request.files["csv_file"]
        # filename = secure_filename(file.filename)
        filename = file.filename
        task_name = f"{ts}_{rs}_{filename}"
        df = pd.read_csv(file)
        # out_path = f"{UPLOAD_FOLDER}/{random_folder}/{filename}.pkl"
        # print(out_path)
        # with open(out_path, "wb") as f:
        #     pickle.dump(df, f)
        # jsonify({"out_path": out_path})
        try:
            clf_modulator_dir = "./man_modulator_115_best_model_20210311-233712.pkl"
            clf_substrate_dir = "./nsc_substrate_mix_103_best_model_20210306-190110.pkl"
            report_d_l = run_pred(df, clf_modulator_dir, clf_substrate_dir)

            return render_template("result.html", items=report_d_l, filename=filename)
        except Exception as e:
            return render_template("error.html", log=e, filename=filename)
    if request.method == "GET":
        return redirect(url_for("wait"))


@app.route("/positive", methods=["GET", "POST"])
def positive():
    with open("./data/sample_pos.csv") as f:
        csv = f.read()
    return Response(
        csv,
        mimetype="text/csv",
        headers={"Content-disposition": "attachment; filename=positive.csv"},
    )


@app.route("/negative", methods=["GET", "POST"])
def negative():
    with open("./data/sample_neg.csv") as f:
        csv = f.read()
    return Response(
        csv,
        mimetype="text/csv",
        headers={"Content-disposition": "attachment; filename=negative.csv"},
    )


if __name__ == "__main__":
    app.run(debug=True)