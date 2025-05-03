# # model_training.py
# import pandas as pd
# import re
# import joblib
# from urllib.parse import urlparse
# from sklearn.model_selection import train_test_split
# from sklearn import svm
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.preprocessing import LabelEncoder

# # --- Feature extraction functions ---
# def contains_ip_address(url):
#     try:
#         match = re.search(r'(([01]?\d\d?|2[0-4]\d|25[0-5])\.'
#                           r'([01]?\d\d?|2[0-4]\d|25[0-5])\.'
#                           r'([01]?\d\d?|2[0-4]\d|25[0-5])\.'
#                           r'([01]?\d\d?|2[0-4]\d|25[0-5]))', url)
#         return 1 if match else 0
#     except:
#         return 0

# def abnormal_url(url):
#     try:
#         hostname = urlparse(url).hostname
#         return 1 if hostname is None else 0
#     except:
#         return 1

# def count_dot(url):
#     try:
#         return url.count('.')
#     except:
#         return 0

# def count_www(url):
#     try:
#         return url.count('www')
#     except:
#         return 0

# def count_atrate(url):
#     try:
#         return url.count('@')
#     except:
#         return 0

# def no_of_dir(url):
#     try:
#         return urlparse(url).path.count('/')
#     except:
#         return 0

# def no_of_embed(url):
#     try:
#         return urlparse(url).path.count('//')
#     except:
#         return 0

# def shortening_service(url):
#     try:
#         match = re.search(r'bit\.ly|goo\.gl|shorte\.st|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|'
#                           r'yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|'
#                           r'short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|'
#                           r'doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|lnkd\.in|db\.tt|qr\.ae|'
#                           r'adf\.ly|cur\.lv|ity\.im|q\.gs|po\.st|bc\.vc|twitthis\.com|u\.to|j\.mp|buzurl\.com|cutt\.us|'
#                           r'u\.bb|yourls\.org|prettylinkpro\.com|scrnch\.me|filoops\.info|vzturl\.com|qr\.net|1url\.com|'
#                           r'tweez\.me|v\.gd|tr\.im|link\.zip\.net', url)
#         return 1 if match else 0
#     except:
#         return 0

# def count_https(url):
#     try:
#         return url.count('https')
#     except:
#         return 0

# def count_http(url):
#     try:
#         return url.count('http')
#     except:
#         return 0

# def count_per(url):
#     try:
#         return url.count('%')
#     except:
#         return 0

# def count_ques(url):
#     try:
#         return url.count('?')
#     except:
#         return 0

# def count_hyphen(url):
#     try:
#         return url.count('-')
#     except:
#         return 0

# def count_equal(url):
#     try:
#         return url.count('=')
#     except:
#         return 0

# def url_length(url):
#     try:
#         return len(str(url))
#     except:
#         return 0

# def hostname_length(url):
#     try:
#         return len(urlparse(url).netloc)
#     except:
#         return 0

# def suspicious_words(url):
#     try:
#         match = re.search(r'PayPal|login|signin|bank|account|update|free|lucky|service|bonus|ebayisapi|webscr', url, re.IGNORECASE)
#         return 1 if match else 0
#     except:
#         return 0

# def digit_count(url):
#     try:
#         return sum(c.isdigit() for c in url)
#     except:
#         return 0

# def letter_count(url):
#     try:
#         return sum(c.isalpha() for c in url)
#     except:
#         return 0

# def fd_length(url):
#     try:
#         path = urlparse(url).path.split('/')
#         return len(path[1]) if len(path) > 1 else 0
#     except:
#         return 0

# # --- Load dataset ---
# df_first = pd.read_csv('dataset.csv', nrows=10000)
# df_last = pd.read_csv('dataset.csv', skiprows=range(1, 450177 - 10000 + 1))  # Adjusted skiprows
# df = pd.concat([df_first, df_last], axis=0).reset_index(drop=True)
# df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# # --- Apply feature extraction ---
# feature_functions = {
#     'use_of_ip': contains_ip_address,
#     'abnormal_url': abnormal_url,
#     'count.': count_dot,
#     'count-www': count_www,
#     'count@': count_atrate,
#     'count_dir': no_of_dir,
#     'count_embed_domian': no_of_embed,
#     'short_url': shortening_service,
#     'count-https': count_https,
#     'count-http': count_http,
#     'count%': count_per,
#     'count?': count_ques,
#     'count-': count_hyphen,
#     'count=': count_equal,
#     'url_length': url_length,
#     'hostname_length': hostname_length,
#     'sus_url': suspicious_words,
#     'fd_length': fd_length,
#     'count-digits': digit_count,
#     'count-letters': letter_count,
# }

# for feature_name, function in feature_functions.items():
#     df[feature_name] = df['url'].apply(lambda url: function(str(url)))

# # --- Encoding labels ---
# lb_make = LabelEncoder()
# df["url_type"] = lb_make.fit_transform(df["type"])

# X = df[list(feature_functions.keys())]
# y = df['url_type']

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # --- Train models ---
# clf_rf = RandomForestClassifier(n_estimators=100, random_state=42)
# clf_rf.fit(X_train, y_train)

# clf_svm = svm.SVC(kernel='linear')
# clf_svm.fit(X_train, y_train)

# # --- Save models ---
# joblib.dump(clf_rf, 'rf_model.pkl')
# joblib.dump(clf_svm, 'svm_model.pkl')
# joblib.dump(lb_make, 'label_encoder.pkl')

# print("Models trained and saved successfully!")
import pandas as pd
import re
import joblib
from urllib.parse import urlparse
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

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

# --- Load dataset ---
df_first = pd.read_csv('dataset.csv', nrows=10000)
df_last = pd.read_csv('dataset.csv', skiprows=range(1, 450177 - 10000 + 1))  # Adjusted skiprows
df = pd.concat([df_first, df_last], axis=0).reset_index(drop=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# --- Apply feature extraction ---
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

for feature_name, function in feature_functions.items():
    df[feature_name] = df['url'].apply(lambda url: function(str(url)))

# --- Encoding labels ---
lb_make = LabelEncoder()
df["url_type"] = lb_make.fit_transform(df["type"])

X = df[list(feature_functions.keys())]
y = df['url_type']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Train models ---
clf_rf = RandomForestClassifier(n_estimators=100, random_state=42)
clf_rf.fit(X_train, y_train)

clf_svm = svm.SVC(kernel='linear')
clf_svm.fit(X_train, y_train)

clf_lr = LogisticRegression(random_state=42)
clf_lr.fit(X_train, y_train)

clf_gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
clf_gb.fit(X_train, y_train)

clf_xgb = XGBClassifier(n_estimators=100, random_state=42)
clf_xgb.fit(X_train, y_train)

# --- Save models ---
joblib.dump(clf_rf, 'rf_model.pkl')
joblib.dump(clf_svm, 'svm_model.pkl')
joblib.dump(clf_lr, 'lr_model.pkl')
joblib.dump(clf_gb, 'gb_model.pkl')
joblib.dump(clf_xgb, 'xgb_model.pkl')
joblib.dump(lb_make, 'label_encoder.pkl')

print("Models trained and saved successfully!")
