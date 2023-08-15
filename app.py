import os
import openai
from dotenv import load_dotenv, find_dotenv
from flask import Flask, request, render_template, session, redirect, url_for
from flask_session import Session

# read env variables from the local .env file
_ = load_dotenv(find_dotenv())
openai.api_key = os.getenv("OPENAI_API_KEY")


def get_chat_response(prompt, model="gpt-3.5-turbo", temperature=1):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model, messages=messages, temperature=temperature
    )
    return response


def try_get_chat_response(prompt, model="gpt-3.5-turbo", temperature=1):
    response = None
    error_message = ""
    try:
        response = get_chat_response(prompt, model, temperature)
    except openai.error.Timeout as e:
        error_message = f"OpenAI API request timed out: {e}"
    except openai.error.APIError as e:
        error_message = f"OpenAI API returned an API Error: {e}"
    except openai.error.APIConnectionError as e:
        error_message = f"OpenAI API request failed to connect: {e}"
    except openai.error.InvalidRequestError as e:
        error_message = f"OpenAI API request was invalid: {e}"
    except openai.error.AuthenticationError as e:
        error_message = f"OpenAI API request was not authorized: {e}"
    except openai.error.PermissionError as e:
        error_message = f"OpenAI API request was not permitted: {e}"
    except openai.error.RateLimitError as e:
        error_message = f"OpenAI API request exceeded rate limit: {e}"
    except Exception as e:
        error_message = f"OpenAI API exception: {e}"

    return response, error_message


app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SESSION_KEY")
app.config["SESSION_TYPE"] = "filesystem"
app.config["SESSION_USE_SIGNER"] = True
Session(app)


@app.route("/chat", methods=["GET", "POST"])
def form_post():
    if "history" not in session:
        session["history"] = []

    if request.method == "POST":
        prompt = request.form["text"]
        model = request.form.get("model")
        temperature = float(request.form.get("temperature"))

        response, error_message = try_get_chat_response(prompt, model, temperature)

        output = (
            error_message
            if response is None
            else response.choices[0].message["content"]
        )

        session["history"].append({"User": prompt, "GPT": output})
        session["model"] = model
        session["temperature"] = temperature

        session.modified = True
        return render_template(
            "home.html",
            history=session["history"],
            model=session["model"],
            temperature=session["temperature"],
        )

    else:
        return render_template(
            "home.html",
            history=session["history"],
            model=session.get("model", "gpt-3.5-turbo"),
            temperature=session.get("temperature", 1),
        )


@app.route("/clear", methods=["POST"])
def clear_history():
    keep = int(request.form.get("keep", 0))

    if keep == 0:
        session["history"] = []
    else:
        # Keep only the last 'keep' messages
        session["history"] = session["history"][-keep:]
    session.modified = True

    # Redirect back to the chat
    return redirect(url_for("form_post"))


if __name__ == "__main__":
    app.run(host="0.0.0.0")
