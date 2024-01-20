import os
from flask import Flask, request, render_template

from predict0 import predict

UPLOAD_FOLDER = "./static/face_images"

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["GET", "POST"])
def upload_user_files():
    if request.method == "POST":
        upload_file = request.files["upload_file"]

        img_path = f"./static/{upload_file.filename}"
        print(img_path)

        upload_file.save(img_path)
        result, score = predict(img_path)
        return render_template(
            "result.html",
            score=int(score * 100),
            result=result,
            img_path="." + img_path,
        )


if __name__ == "__main__":
    app.run(debug=True)
