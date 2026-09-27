
from flask import Flask, render_template, request, session
from difflib import SequenceMatcher
import pickle
import re
from urllib.parse import urlparse
import string
import os

app = Flask(__name__)
app.secret_key = "phishguard-local-dev-key"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SMS_MODEL_PATH = os.path.join(BASE_DIR, "sms_model.pkl")
SMS_VECTORIZER_PATH = os.path.join(BASE_DIR, "sms_vectorizer.pkl")

vector = pickle.load(open("vectorizer.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))

def load_sms_model():
    try:
        if not os.path.exists(SMS_MODEL_PATH) or not os.path.exists(SMS_VECTORIZER_PATH):
            return None, None
        vec = pickle.load(open(SMS_VECTORIZER_PATH, "rb"))
        mod = pickle.load(open(SMS_MODEL_PATH, "rb"))
        return vec, mod
    except (FileNotFoundError, OSError, pickle.PickleError):
        return None, None

sms_vectorizer, sms_model = load_sms_model()

SUSPICIOUS_SMS_TOKENS = ["win", "prize", "free", "urgent", "click", "bank", "kyc", "otp", "verify", "reward", "offer", "loan"]


BAD_KEYWORDS = ("verify", "account", "secure", "login", "update", "bank", "password", "paypal")
IP_PATTERN = re.compile(r"^(?:\d{1,3}\.){3}\d{1,3}$")
URL_PATTERN = re.compile(r"(?:https?://|www\.)[^\s<>()\"']+|[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?:/[^\s<>()\"']*)?")
KNOWN_BRANDS = ("paypal", "google", "amazon", "microsoft", "apple", "facebook", "instagram", "netflix", "bank")
COMMON_TLDS = {"com", "org", "net", "edu", "gov", "in", "co", "io", "ai"}
LEET_MAP = str.maketrans({"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"})


def clean_message_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def get_url_parts(url):
    parsed = urlparse(url if re.match(r"^[a-z]+://", url, re.I) else f"http://{url}")
    host = parsed.netloc.split("@")[-1].split(":")[0].lower()
    path = parsed.path.lower()
    return parsed, host, path


def get_registered_domain(host):
    host_parts = [part for part in host.split(".") if part]
    if len(host_parts) >= 2:
        return ".".join(host_parts[-2:])
    return host


def extract_urls_from_text(text):
    urls = []
    for match in URL_PATTERN.findall(text or ""):
        cleaned = match.rstrip(".,;:!?)]}")
        if "." in cleaned and cleaned not in urls:
            urls.append(cleaned)
    return urls


def detect_brand_spoof(url):
    _, host, _ = get_url_parts(url)
    domain = get_registered_domain(host)
    domain_name = domain.split(".")[0]
    normalized = domain_name.translate(LEET_MAP)

    best_brand = None
    best_score = 0
    for brand in KNOWN_BRANDS:
        ratio = SequenceMatcher(None, normalized, brand).ratio()
        contains_brand = brand in normalized and normalized != brand
        leet_overlap = domain_name != normalized and SequenceMatcher(None, normalized, brand).ratio() > 0.72
        score = max(ratio, 0.88 if contains_brand else 0, 0.82 if leet_overlap else 0)
        if score > best_score:
            best_score = score
            best_brand = brand

    if best_score >= 0.82 and normalized != best_brand:
        return {
            "brand": best_brand.title(),
            "score": round(best_score * 100),
            "risk": "Possible brand spoof",
            "details": f"{domain_name} is lexically similar to {best_brand}",
            "status": "bad",
        }

    return {
        "brand": "None",
        "score": round(best_score * 100) if best_score else 0,
        "risk": "No close brand spoof pattern",
        "details": "No high-similarity brand token detected",
        "status": "good",
    }


def get_domain_info(url):
    parsed, host, _ = get_url_parts(url)
    domain = get_registered_domain(host)
    domain_parts = domain.split(".")
    tld = domain_parts[-1] if len(domain_parts) > 1 else "unknown"
    subdomain_count = max(len([part for part in host.split(".") if part]) - 2, 0)

    return {
        "domain_age": "Unavailable in offline demo",
        "domain_length": len(domain),
        "tld_type": f".{tld} ({'common' if tld in COMMON_TLDS else 'less common'})",
        "https_presence": "Present" if parsed.scheme == "https" else "Missing",
        "subdomain_count": subdomain_count,
        "registrar": "Not available locally",
        "whois_availability": "WHOIS lookup not configured",
    }


def generate_model_recommendations(prediction):
    if prediction.get("status") == "danger":
        return [
            "High lexical anomaly detected by URL pattern features.",
            "Verify the source manually before trusting the domain.",
            "Avoid credential submission until the domain is cross-checked.",
            "Cross-check the registered domain, TLD, and suspicious tokens.",
        ]

    if prediction.get("status") == "safe":
        return [
            "Low lexical anomaly for the submitted URL pattern.",
            "Standard domain pattern observed by feature extraction.",
            "Continue with caution, especially if the URL came from an unexpected message.",
            "Use domain context and source reliability alongside the ML prediction.",
        ]

    return [
        "Model confidence context is limited for this input.",
        "Inspect lexical features and domain characteristics manually.",
        "Use multiple signals before making a final decision.",
    ]


def analyze_url_features(url, status=None):
    if not url:
        return {
            "risk_score": 0,
            "feature_scores": [],
            "reasons": [],
            "features": [
                {"name": "HTTPS", "value": "-", "status": "neutral"},
                {"name": "IP in URL", "value": "-", "status": "neutral"},
                {"name": "Subdomains", "value": "-", "status": "neutral"},
                {"name": "Bad keywords", "value": "-", "status": "neutral"},
                {"name": "URL length", "value": "-", "status": "neutral"},
            ],
        }

    parsed, host, path = get_url_parts(url)
    host_parts = [part for part in host.split(".") if part]
    subdomains = max(len(host_parts) - 2, 0)
    token_text = f"{host} {path}"
    suspicious_tokens = [keyword for keyword in BAD_KEYWORDS if keyword in token_text]
    bad_keyword_found = bool(suspicious_tokens)
    has_ip = bool(IP_PATTERN.match(host))
    has_https = parsed.scheme == "https"
    url_length = len(url)
    suspicious_symbols = len(re.findall(r"[-_%@=]", url))
    brand_spoof = detect_brand_spoof(url)

    feature_points = 0
    feature_scores = [
        {"feature": "HTTPS", "value": "Missing" if not has_https else "Present", "contribution": 18 if not has_https else 0},
        {"feature": "URL Length", "value": "High" if url_length > 75 else "Normal", "contribution": 12 if url_length > 75 else 0},
        {"feature": "Suspicious Tokens", "value": ", ".join(suspicious_tokens) if suspicious_tokens else "None", "contribution": min(len(suspicious_tokens) * 8, 24)},
        {"feature": "Subdomain Count", "value": f"{subdomains} levels", "contribution": min(subdomains * 8, 20)},
        {"feature": "Brand Similarity Risk", "value": brand_spoof["details"], "contribution": 18 if brand_spoof["status"] == "bad" else 0},
        {"feature": "Suspicious Symbols", "value": str(suspicious_symbols), "contribution": min(suspicious_symbols * 3, 12)},
        {"feature": "IP Address Use", "value": "Present" if has_ip else "None", "contribution": 20 if has_ip else 0},
    ]
    feature_points = sum(item["contribution"] for item in feature_scores)

    if status == "danger":
        score = max(55, min(100, feature_points + 35))
    elif status == "safe":
        score = min(25, feature_points // 3)
    else:
        score = min(100, feature_points)

    reasons = [item for item in feature_scores if item["contribution"] > 0]
    if not reasons:
        reasons = [{"feature": "Lexical Pattern", "value": "Standard URL structure", "contribution": 0}]

    return {
        "risk_score": score,
        "feature_scores": feature_scores,
        "reasons": reasons,
        "features": [
            {"name": "HTTPS", "value": "Present" if has_https else "Missing", "status": "good" if has_https else "bad"},
            {"name": "IP in URL", "value": "Present" if has_ip else "None", "status": "bad" if has_ip else "good"},
            {"name": "Subdomains", "value": f"{subdomains} levels", "status": "bad" if subdomains > 2 else "good"},
            {"name": "Bad keywords", "value": ", ".join(suspicious_tokens) if suspicious_tokens else "None", "status": "bad" if bad_keyword_found else "good"},
            {"name": "URL length", "value": f"{url_length} chars", "status": "bad" if url_length > 75 else "good"},
        ],
    }


def classify_url(url):
    cleaned_url = re.sub(r'^https?://(www\.)?', '', url.strip())
    prediction = model.predict(vector.transform([cleaned_url]))[0]

    if prediction == "bad":
        return {
            "status": "danger",
            "label": "This is a Phishing Website !!",
            "message": "This URL matches patterns commonly seen in phishing websites.",
        }

    if prediction == "good":
        return {
            "status": "safe",
            "label": "This is a Legitimate Website !!",
            "message": "This URL looks legitimate according to the trained model.",
        }

    return {
        "status": "unknown",
        "label": "Unknown prediction",
        "message": "The model could not classify this URL confidently.",
    }


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    submitted_url = ""
    history = session.get("scan_history", [])
    analysis = analyze_url_features("")
    brand_spoof = detect_brand_spoof("example.com")
    domain_info = get_domain_info("example.com")
    recommendations = generate_model_recommendations({"status": "unknown"})

    if request.method == "POST":
        submitted_url = request.form.get("url", "").strip()

        if not submitted_url:
            result = {
                "status": "unknown",
                "label": "Enter a URL",
                "message": "Please enter a valid website URL before scanning.",
            }
            return render_template("index.html", active_tab="url", result=result, submitted_url=submitted_url, history=history, analysis=analysis, stats=get_stats(history), brand_spoof=brand_spoof, domain_info=domain_info, recommendations=recommendations)

        result = classify_url(submitted_url)
        analysis = analyze_url_features(submitted_url, result["status"])
        brand_spoof = detect_brand_spoof(submitted_url)
        domain_info = get_domain_info(submitted_url)
        recommendations = generate_model_recommendations(result)
        history_entry = {
            "url": submitted_url,
            "status": result["status"],
            "label": result["label"],
            "risk_score": analysis["risk_score"],
        }
        history = [history_entry] + [item for item in history if item.get("url") != submitted_url]
        history = history[:8]
        session["scan_history"] = history

    return render_template("index.html", active_tab="url", result=result, submitted_url=submitted_url, history=history, analysis=analysis, stats=get_stats(history), brand_spoof=brand_spoof, domain_info=domain_info, recommendations=recommendations)


def predict_message_text(text):
    if not sms_model or not sms_vectorizer:
        return {"prediction": "unknown", "confidence": 0}

    cleaned_text = clean_message_text(text)
    if not cleaned_text:
        return {"prediction": "unknown", "confidence": 0}

    vec = sms_vectorizer.transform([cleaned_text])
    pred = sms_model.predict(vec)[0]

    try:
        proba = sms_model.predict_proba(vec)[0]
        confidence = round(max(proba) * 100)
    except Exception:
        confidence = 0

    return {"prediction": pred, "confidence": confidence}


def detect_suspicious_tokens(text):
    text_lower = text.lower()
    return [token for token in SUSPICIOUS_SMS_TOKENS if token in text_lower]


def calculate_message_risk_score(message_pred_info, suspicious_tokens, url_results, text):
    score = 0

    # Spam probability = 40%
    if message_pred_info["prediction"] == "spam":
        score += min(40, (message_pred_info["confidence"] / 100.0) * 40)
    elif message_pred_info["prediction"] == "ham":
        score += max(0, 10 - (message_pred_info["confidence"] / 100.0) * 10)

    # Suspicious token count = 25% (e.g. 5% per token)
    score += min(25, len(suspicious_tokens) * 5)

    # URL count = 15%
    if len(url_results) == 1:
        score += 10
    elif len(url_results) > 1:
        score += 15

    # Existing URL phishing predictions = 20%
    if url_results:
        max_url_risk = max([res["risk_score"] for res in url_results])
        score += (max_url_risk / 100.0) * 20

    final_score = round(min(100, score))

    if final_score <= 25:
        label = "Low"
    elif final_score <= 50:
        label = "Moderate"
    elif final_score <= 75:
        label = "High"
    else:
        label = "Critical"

    return final_score, label


def analyze_text_submission(text):
    pred_info = predict_message_text(text)
    urls = extract_urls_from_text(text)
    
    url_results = []
    for url in urls:
        prediction = classify_url(url)
        analysis = analyze_url_features(url, prediction["status"])
        url_results.append({
            "url": url,
            "prediction": prediction["status"],
            "risk_score": analysis["risk_score"],
            "key_features": [item for item in analysis["reasons"][:3]]
        })
        
    tokens = detect_suspicious_tokens(text)
    risk_score, risk_label = calculate_message_risk_score(pred_info, tokens, url_results, text)
    
    return {
        "message_prediction": pred_info["prediction"].capitalize(),
        "confidence_score": pred_info["confidence"],
        "message_risk_score": risk_score,
        "risk_label": risk_label,
        "suspicious_tokens": tokens,
        "urls_found": urls,
        "url_results": url_results
    }


@app.route("/text-link-analyzer", methods=["GET", "POST"])
def text_link_analyzer():
    message_text = ""
    analysis_result = None
    error_message = None
    info_message = None

    if request.method == "POST":
        message_text = request.form.get("message_text", "").strip()

        if not message_text:
            error_message = "Please enter message text"
        elif not sms_model or not sms_vectorizer:
            error_message = "SMS model not trained. Run train_sms_model.py"
        else:
            analysis_result = analyze_text_submission(message_text)
            if not analysis_result["urls_found"]:
                info_message = "No links detected — text classification only"

    return render_template("text_link_analyzer.html", 
                           active_tab="text", 
                           message_text=message_text, 
                           analysis_result=analysis_result,
                           error_message=error_message,
                           info_message=info_message)


@app.route("/model-insights")
def model_insights():
    history = session.get("scan_history", [])
    return render_template("model_insights.html", active_tab="insights", history=history, stats=get_stats(history))


def get_stats(history):
    checked = len(history)
    safe = sum(1 for item in history if item.get("status") == "safe")
    phishing = sum(1 for item in history if item.get("status") == "danger")
    threat_rate = round((phishing / checked) * 100) if checked else 0
    return {
        "checked": checked,
        "safe": safe,
        "phishing": phishing,
        "threat_rate": threat_rate,
    }


if __name__ == "__main__":
    app.run(debug=False, port=5001)
