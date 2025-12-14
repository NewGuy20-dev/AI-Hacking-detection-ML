import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import pickle

def preprocess_data(df):
    """Encode categorical features"""
    le_dict = {}
    categorical_cols = ['protocol_type', 'service', 'flag']
    
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le
    
    # Select features (exclude target and non-feature columns)
    feature_cols = [col for col in df.columns if col not in ['attack_type', 'is_attack', 'difficulty']]
    return df[feature_cols], le_dict

def train_model():
    """Train hacking detection model"""
    # Load data
    train_df = pd.read_csv('data/nsl_kdd_train.csv', names=[
        'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
        'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
        'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
        'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
        'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
        'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
        'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
        'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
        'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
        'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'attack_type', 'difficulty'
    ])
    
    train_df['is_attack'] = (train_df['attack_type'] != 'normal').astype(int)
    
    # Preprocess
    X_train, le_dict = preprocess_data(train_df)
    y_train = train_df['is_attack']
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Save model and encoders
    with open('model.pkl', 'wb') as f:
        pickle.dump({'model': model, 'encoders': le_dict}, f)
    
    return model, X_train.columns.tolist()

def evaluate_model():
    """Evaluate model on test data"""
    # Load test data
    test_df = pd.read_csv('data/nsl_kdd_test.csv', names=[
        'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
        'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
        'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
        'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
        'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
        'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
        'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
        'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
        'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
        'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'attack_type', 'difficulty'
    ])
    
    test_df['is_attack'] = (test_df['attack_type'] != 'normal').astype(int)
    
    # Load model
    with open('model.pkl', 'rb') as f:
        saved_data = pickle.load(f)
        model = saved_data['model']
        le_dict = saved_data['encoders']
    
    # Preprocess test data
    for col, le in le_dict.items():
        test_df[col] = le.transform(test_df[col])
    
    feature_cols = [col for col in test_df.columns if col not in ['attack_type', 'is_attack', 'difficulty']]
    X_test = test_df[feature_cols]
    y_test = test_df['is_attack']
    
    # Predict
    y_pred = model.predict(X_test)
    
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Attack']))

if __name__ == "__main__":
    print("Training model...")
    model, features = train_model()
    print("Evaluating model...")
    evaluate_model()
