from flask import Flask, request, jsonify, render_template, redirect, url_for, session
import joblib
import sqlite3
import os

app = Flask(__name__)

# Secret key for login sessions
app.secret_key = "fake-job-detection-secret-key"

# --------------------------------------------------
# LOAD MACHINE LEARNING MODEL
# --------------------------------------------------

model = joblib.load("fake_job_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

print("Model loaded successfully!")
print("Vectorizer loaded successfully!")


# --------------------------------------------------
# DATABASE
# --------------------------------------------------

DATABASE = "users.db"


def get_db_connection():
    connection = sqlite3.connect(DATABASE)
    connection.row_factory = sqlite3.Row
    return connection


def create_database():

    connection = get_db_connection()

    connection.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fullname TEXT NOT NULL,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    """)

    connection.commit()
    connection.close()


# Create database when application starts
create_database()


# --------------------------------------------------
# FAKE JOB PREDICTION
# --------------------------------------------------

def predict_job_posting(job_description, job_url=""):

    text = f"{job_description} {job_url}"

    text_tfidf = vectorizer.transform([text])

    prediction = model.predict(text_tfidf)[0]

    probabilities = model.predict_proba(text_tfidf)[0]

    confidence = max(probabilities) * 100

    if prediction == 1:
        result = "Fake"
        status = "Fraudulent"
    else:
        result = "Genuine"
        status = "Safe"

    return {
        "prediction": result,
        "confidence": round(confidence, 2),
        "status": status
    }


# --------------------------------------------------
# LOGIN
# --------------------------------------------------

@app.route("/", methods=["GET", "POST"])
def login():

    if session.get("logged_in"):
        return redirect(url_for("dashboard"))

    if request.method == "POST":

        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        connection = get_db_connection()

        user = connection.execute(
            "SELECT * FROM users WHERE username = ? AND password = ?",
            (username, password)
        ).fetchone()

        connection.close()

        if user:

            session["logged_in"] = True
            session["username"] = user["username"]
            session["fullname"] = user["fullname"]

            return redirect(url_for("dashboard"))

        return render_template(
            "login.html",
            error="Invalid username or password."
        )

    return render_template("login.html")


# --------------------------------------------------
# REGISTRATION
# --------------------------------------------------

@app.route("/register", methods=["GET", "POST"])
def register():

    if session.get("logged_in"):
        return redirect(url_for("dashboard"))

    if request.method == "POST":

        fullname = request.form.get("fullname", "").strip()
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        # Check empty fields
        if not fullname or not username or not email or not password:

            return render_template(
                "register.html",
                error="Please fill all fields."
            )

        # Check password
        if password != confirm_password:

            return render_template(
                "register.html",
                error="Passwords do not match."
            )

        # Minimum password length
        if len(password) < 5:

            return render_template(
                "register.html",
                error="Password must contain at least 5 characters."
            )

        connection = get_db_connection()

        try:

            connection.execute(
                """
                INSERT INTO users
                (fullname, username, email, password)
                VALUES (?, ?, ?, ?)
                """,
                (fullname, username, email, password)
            )

            connection.commit()
            connection.close()

            return render_template(
                "login.html",
                success="Registration successful! Please login."
            )

        except sqlite3.IntegrityError:

            connection.close()

            return render_template(
                "register.html",
                error="Username or email already exists."
            )

    return render_template("register.html")


# --------------------------------------------------
# FORGOT PASSWORD
# --------------------------------------------------

@app.route("/forgot-password", methods=["GET", "POST"])
def forgot_password():

    if request.method == "POST":

        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip()
        new_password = request.form.get("new_password", "")
        confirm_password = request.form.get("confirm_password", "")

        if not username or not email or not new_password:

            return render_template(
                "forgot_password.html",
                error="Please fill all fields."
            )

        if new_password != confirm_password:

            return render_template(
                "forgot_password.html",
                error="Passwords do not match."
            )

        if len(new_password) < 5:

            return render_template(
                "forgot_password.html",
                error="Password must contain at least 5 characters."
            )

        connection = get_db_connection()

        user = connection.execute(
            "SELECT * FROM users WHERE username = ? AND email = ?",
            (username, email)
        ).fetchone()

        if not user:

            connection.close()

            return render_template(
                "forgot_password.html",
                error="Username and email do not match."
            )

        connection.execute(
            "UPDATE users SET password = ? WHERE username = ? AND email = ?",
            (new_password, username, email)
        )

        connection.commit()
        connection.close()

        return render_template(
            "login.html",
            success="Password reset successfully! Please login."
        )

    return render_template("forgot_password.html")


# --------------------------------------------------
# DASHBOARD
# --------------------------------------------------

@app.route("/dashboard")
def dashboard():

    if not session.get("logged_in"):
        return redirect(url_for("login"))

    return render_template(
        "dashboard.html",
        username=session.get("username"),
        fullname=session.get("fullname")
    )


# --------------------------------------------------
# FAKE JOB DETECTION PAGE
# --------------------------------------------------

@app.route("/detect")
def detect():

    if not session.get("logged_in"):
        return redirect(url_for("login"))

    return render_template("index.html")


# --------------------------------------------------
# ANALYZE JOB API
# --------------------------------------------------

@app.route("/api/analyze", methods=["POST"])
def analyze():

    if not session.get("logged_in"):

        return jsonify({
            "error": "Please login first."
        }), 401

    try:

        data = request.get_json()

        if not data:

            return jsonify({
                "error": "Invalid request."
            }), 400

        job_description = data.get(
            "job_description",
            ""
        )

        job_url = data.get(
            "job_url",
            ""
        )

        if not job_description and not job_url:

            return jsonify({
                "error": "Please enter a job description."
            }), 400

        result = predict_job_posting(
            job_description,
            job_url
        )

        return jsonify(result)

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


# --------------------------------------------------
# LOGOUT
# --------------------------------------------------

@app.route("/logout")
def logout():

    session.clear()

    return redirect(url_for("login"))


# --------------------------------------------------
# START APPLICATION
# --------------------------------------------------

if __name__ == "__main__":

    print("--------------------------------------")
    print(" Fake Job Detection Using ML")
    print("--------------------------------------")
    print("Server starting...")
    print("Open http://127.0.0.1:5000")

    app.run(
        debug=True,
        port=5000
    )