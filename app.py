from flask import Flask, request, jsonify
import os
from flask_cors import CORS
import joblib
import re
import numpy as np
from urllib.parse import urlparse
from googlesearch import search
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

app = Flask(__name__)
CORS(app)

# --- Load Models ---
clf_rf = joblib.load('rf_model.pkl')
clf_svm = joblib.load('svm_model.pkl')
clf_lr = joblib.load('lr_model.pkl')  # Load Logistic Regression Model
clf_gb = joblib.load('gb_model.pkl')  # Load Gradient Boosting Model
clf_xgb = joblib.load('xgb_model.pkl')  # Load XGBoost Model
lb_make = joblib.load('label_encoder.pkl')

# --- Feature extraction functions ---
def contains_ip_address(url):
    try:
        match = re.search(r'(([01]?\d\d?|2[0-4]\d|25[0-5])\.'
                          r'([01]?\d\d?|2[0-4]\d|25[0-5])\.'
                          r'([01]?\d\d?|2[0-4]\d|25[0-5])\.'
                          r'([01]?\d\d?|2[0-4]\d|25[0-5]))', url)
        return 1 if match else 0
    except:
        return 0

def abnormal_url(url):
    try:
        hostname = urlparse(url).hostname
        return 1 if hostname is None else 0
    except:
        return 1

def count_dot(url):
    try:
        return url.count('.')
    except:
        return 0

def count_www(url):
    try:
        return url.count('www')
    except:
        return 0

def count_atrate(url):
    try:
        return url.count('@')
    except:
        return 0

def no_of_dir(url):
    try:
        return urlparse(url).path.count('/')
    except:
        return 0

def no_of_embed(url):
    try:
        return urlparse(url).path.count('//')
    except:
        return 0

def shortening_service(url):
    try:
        match = re.search(r'bit\.ly|goo\.gl|shorte\.st|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|'
                          r'yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|'
                          r'short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|'
                          r'doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|lnkd\.in|db\.tt|qr\.ae|'
                          r'adf\.ly|cur\.lv|ity\.im|q\.gs|po\.st|bc\.vc|twitthis\.com|u\.to|j\.mp|buzurl\.com|cutt\.us|'
                          r'u\.bb|yourls\.org|prettylinkpro\.com|scrnch\.me|filoops\.info|vzturl\.com|qr\.net|1url\.com|'
                          r'tweez\.me|v\.gd|tr\.im|link\.zip\.net', url)
        return 1 if match else 0
    except:
        return 0

def count_https(url):
    try:
        return url.count('https')
    except:
        return 0

def count_http(url):
    try:
        return url.count('http')
    except:
        return 0

def count_per(url):
    try:
        return url.count('%')
    except:
        return 0

def count_ques(url):
    try:
        return url.count('?')
    except:
        return 0

def count_hyphen(url):
    try:
        return url.count('-')
    except:
        return 0

def count_equal(url):
    try:
        return url.count('=')
    except:
        return 0

def url_length(url):
    try:
        return len(str(url))
    except:
        return 0

def hostname_length(url):
    try:
        return len(urlparse(url).netloc)
    except:
        return 0

def suspicious_words(url):
    try:
        match = re.search(r'PayPal|login|signin|bank|account|update|free|lucky|service|bonus|ebayisapi|webscr', url, re.IGNORECASE)
        return 1 if match else 0
    except:
        return 0

def digit_count(url):
    try:
        return sum(c.isdigit() for c in url)
    except:
        return 0

def letter_count(url):
    try:
        return sum(c.isalpha() for c in url)
    except:
        return 0

def fd_length(url):
    try:
        path = urlparse(url).path.split('/')
        return len(path[1]) if len(path) > 1 else 0
    except:
        return 0

feature_functions = {
    'use_of_ip': contains_ip_address,
    'abnormal_url': abnormal_url,
    'count.': count_dot,
    'count-www': count_www,
    'count@': count_atrate,
    'count_dir': no_of_dir,
    'count_embed_domian': no_of_embed,
    'short_url': shortening_service,
    'count-https': count_https,
    'count-http': count_http,
    'count%': count_per,
    'count?': count_ques,
    'count-': count_hyphen,
    'count=': count_equal,
    'url_length': url_length,
    'hostname_length': hostname_length,
    'sus_url': suspicious_words,
    'fd_length': fd_length,
    'count-digits': digit_count,
    'count-letters': letter_count,
}

# --- Prediction Functions ---
def extract_features(url):
    return [func(str(url)) for func in feature_functions.values()]

def predict_url_type(url):
    features = extract_features(url)

    # Predictions
    prediction_rf = clf_rf.predict([features])[0]
    confidence_rf = clf_rf.predict_proba([features])[0][prediction_rf]

    prediction_svm = clf_svm.predict([features])[0]
    try:
        if hasattr(clf_svm, "predict_proba"):
            confidence_svm = clf_svm.predict_proba([features])[0][prediction_svm]
        else:
            raw_score = clf_svm.decision_function([features])[0]
            confidence_svm = 1 / (1 + np.exp(-np.clip(raw_score, -100, 100)))
    except:
        confidence_svm = 0.5

    prediction_lr = clf_lr.predict([features])[0]
    confidence_lr = clf_lr.predict_proba([features])[0][prediction_lr]

    prediction_gb = clf_gb.predict([features])[0]
    confidence_gb = clf_gb.predict_proba([features])[0][prediction_gb]

    prediction_xgb = clf_xgb.predict([features])[0]
    confidence_xgb = clf_xgb.predict_proba([features])[0][prediction_xgb]

    # Decode labels
    prediction_rf_label = lb_make.inverse_transform([prediction_rf])[0]
    prediction_svm_label = lb_make.inverse_transform([prediction_svm])[0]
    prediction_lr_label = lb_make.inverse_transform([prediction_lr])[0]
    prediction_gb_label = lb_make.inverse_transform([prediction_gb])[0]
    prediction_xgb_label = lb_make.inverse_transform([prediction_xgb])[0]

    # Updated explanation and risk level
    explanation, risk_level = generate_explanation(features, prediction_rf_label)

    return {
        "prediction_rf": prediction_rf_label,
        "confidence_rf": confidence_rf,
        "prediction_svm": prediction_svm_label,
        "confidence_svm": confidence_svm,
        "prediction_lr": prediction_lr_label,
        "confidence_lr": confidence_lr,
        "prediction_gb": prediction_gb_label,
        "confidence_gb": confidence_gb,
        "prediction_xgb": prediction_xgb_label,
        "confidence_xgb": confidence_xgb,
        "explanation": explanation,
        "risk_level": risk_level
    }


# --- Updated generate_explanation ---
def generate_explanation(features, prediction):
    explanation = []

    # Feature mappings for clarity
    (
        use_of_ip, abnormal_url, count_dot, count_www, count_atrate, count_dir,
        count_embed_domain, short_url, count_https, count_http, count_percent,
        count_question, count_hyphen, count_equal, url_len, hostname_len,
        suspicious_word, fd_len, count_digits, count_letters
    ) = features

    # Analyze extracted features
    if use_of_ip:
        explanation.append("The URL uses an IP address instead of a domain name, which is suspicious.")
    if abnormal_url:
        explanation.append("The URL is malformed or abnormal, indicating possible phishing.")
    if count_dot > 4:
        explanation.append("The URL has too many dots, suggesting a complicated and potentially deceptive structure.")
    if count_www == 0:
        explanation.append("URL does not contain 'www', which is unusual for legitimate sites.")
    if count_atrate > 0:
        explanation.append("The URL contains '@' symbol, which can mislead users into thinking they're on a legitimate site.")
    if count_embed_domain > 0:
        explanation.append("The URL has multiple embedded domains ('//'), a common trick in phishing.")
    if short_url:
        explanation.append("The URL uses a known URL shortening service, which can hide the final destination.")
    if count_hyphen > 3:
        explanation.append("Excessive hyphens in the domain or path, which may indicate a fake website.")
    if suspicious_word:
        explanation.append("Suspicious keywords like 'login', 'bank', 'update' detected in the URL.")
    if count_question > 2 or count_equal > 3:
        explanation.append("Multiple query parameters ('?' or '=') found, could indicate phishing.")
    if url_len > 75:
        explanation.append("The URL is excessively long, often used to obscure malicious content.")
    if hostname_len > 50:
        explanation.append("The hostname is very long, which is unusual for trustworthy sites.")
    if fd_len > 15:
        explanation.append("The first directory name is suspiciously long.")

    # Set Risk Level based on severity
    if prediction == "phishing":
        if len(explanation) >= 5:
            risk_level = "High Risk"
        elif len(explanation) >= 3:
            risk_level = "Moderate Risk"
        else:
            risk_level = "Low Risk"
        explanation.append("Machine learning model predicts this URL as malicious.")
    else:
        if len(explanation) >= 3:
            risk_level = "Moderate Risk"
            explanation.append("Some features are concerning, but model predicts the URL as legitimate.")
        else:
            risk_level = "Safe"
            explanation.append("Features and model both suggest the URL is safe.")

    return explanation, risk_level



def google_search(url):
    try:
        parsed_url = urlparse(url)
        domain = parsed_url.netloc.split(':')[0]

        query = f"site:{domain}"

        # Updated to match googlesearch-python
        results = search(query, num_results=5)

        return list(results)
    except Exception as e:
        print(f"Error during Google search: {e}")
        return []

# --- CSP Headers ---
@app.after_request
def add_csp_headers(response):
    response.headers['Content-Security-Policy'] = "script-src 'self' 'unsafe-inline'; object-src 'self'"
    return response

# --- API Endpoints ---
@app.route('/')
def home():
    return 'Hello, Flask is running!'

@app.route('/ping')
def ping():
    return "I'm awake!", 200

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    url = data.get('url')

    if not url:
        return jsonify({'error': 'No URL provided'}), 400

    result = predict_url_type(url)

    # Gather predictions
    prediction_rf = result['prediction_rf']
    confidence_rf = round(result['confidence_rf'] * 100, 2)
    prediction_svm = result['prediction_svm']
    confidence_svm = round(result['confidence_svm'] * 100, 2)
    prediction_lr = result['prediction_lr']
    confidence_lr = round(result['confidence_lr'] * 100, 2)
    prediction_gb = result['prediction_gb']
    confidence_gb = round(result['confidence_gb'] * 100, 2)
    prediction_xgb = result['prediction_xgb']
    confidence_xgb = round(result['confidence_xgb'] * 100, 2)

    explanation = result['explanation']
    risk_level = result['risk_level']

    google_results = google_search(url)

    result_str = "✅ URL IS SAFE!" if prediction_rf == 'legitimate' else "⚠️ URL IS MALICIOUS!"

    return jsonify({
        'prediction_rf': prediction_rf,
        'confidence_rf': f"{confidence_rf}%",
        'prediction_svm': prediction_svm,
        'confidence_svm': f"{confidence_svm}%",
        'prediction_lr': prediction_lr,
        'confidence_lr': f"{confidence_lr}%",
        'prediction_gb': prediction_gb,
        'confidence_gb': f"{confidence_gb}%",
        'prediction_xgb': prediction_xgb,
        'confidence_xgb': f"{confidence_xgb}%",
        'explanation': explanation,
        'risk_level': risk_level,
        'result_str': result_str,
        'google_results': google_results
    })

if __name__ == '__main__':
   port = int(os.environ.get("PORT", 5000))  # Use the PORT env variable or default to 5000
   app.run(host="0.0.0.0", port=port)

