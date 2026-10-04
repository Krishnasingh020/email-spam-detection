import os
import random
import requests
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
DATASET_PATH = DATA_DIR / "dataset.csv"

def try_load_kaggle() -> pd.DataFrame | None:
    """Attempt to download dataset using kagglehub if available and configured."""
    try:
        import kagglehub
        # Try primary dataset
        for dataset_id in [
            "purusinghvi/email-spam-classification-dataset",
            "bayes2003/emails-for-spam-or-ham-classification-enron-2006"
        ]:
            try:
                path = kagglehub.dataset_download(dataset_id)
                csv_files = list(Path(path).glob("*.csv"))
                if csv_files:
                    df = pd.read_csv(csv_files[0], encoding_errors="replace")
                    cleaned_df = _standardize_columns(df)
                    if cleaned_df is not None and len(cleaned_df) >= 500:
                        print(f"[DataLoader] Loaded via Kaggle ({dataset_id})")
                        return cleaned_df
            except Exception:
                continue
    except (ImportError, Exception):
        pass
    return None

def try_load_github_mirror() -> pd.DataFrame | None:
    """Attempt direct HTTP GET of public email/spam CSV/TSV datasets with 10s timeout."""
    mirrors = [
        # SMS Spam Collection (TSV format: label \t text)
        {
            "url": "https://raw.githubusercontent.com/justmarkham/DAT8/master/data/sms.tsv",
            "sep": "\t",
            "names": ["label", "text"],
            "encoding": "utf-8"
        },
        # Kaggle SMS Spam Collection mirror (v1, v2)
        {
            "url": "https://raw.githubusercontent.com/mohitgupta-omg/Kaggle-SMS-Spam-Collection-Dataset-csv/master/spam.csv",
            "sep": ",",
            "encoding": "latin-1"
        },
        # Susanli spam dataset mirror
        {
            "url": "https://raw.githubusercontent.com/susanli2016/Machine-Learning-with-Python/master/data/spam.csv",
            "sep": ",",
            "encoding": "latin-1"
        }
    ]

    for mirror in mirrors:
        try:
            resp = requests.get(mirror["url"], timeout=10)
            if resp.status_code == 200:
                from io import StringIO
                df = pd.read_csv(
                    StringIO(resp.text),
                    sep=mirror.get("sep", ","),
                    names=mirror.get("names", None),
                    encoding_errors="replace"
                )
                cleaned = _standardize_columns(df)
                if cleaned is not None and len(cleaned) >= 500:
                    print(f"[DataLoader] Loaded via GitHub mirror ({mirror['url']})")
                    return cleaned
        except Exception:
            continue
    return None

def generate_synthetic_dataset(num_samples: int = 2400) -> pd.DataFrame:
    """
    Final guaranteed fallback: Generates a realistic labeled dataset of 2,000+ rows
    combining varied templates and domain-specific vocabulary banks.
    """
    random.seed(42)

    spam_intros = [
        "URGENT NOTICE:", "Congratulations!", "You have won!", "Final Warning:",
        "Special Promotion!", "Act Now:", "Exclusive Invitation:", "Dear Valued Customer,",
        "Attention Account Holder:", "Instant Approval:", "Claim Your Prize:", "Limited Offer:"
    ]
    spam_actions = [
        "claim your $1,000,000 lottery jackpot right now",
        "verify your credit card details immediately to avoid account suspension",
        "order discounted generic viagra and pharmaceuticals with 80% off",
        "apply for a pre-approved risk-free personal loan with zero interest",
        "click here to receive your guaranteed reward cash prize",
        "unlock your free gift card by submitting your personal information",
        "act now before this once-in-a-lifetime investment opportunity expires",
        "earn $5,000 weekly working from home with no prior experience",
        "redeem your complimentary luxury cruise voucher today",
        "secure your crypto wallet before your access is terminated forever"
    ]
    spam_urgency = [
        "This exclusive offer expires in 24 hours.", "Do not delay or lose your funds.",
        "Reply immediately with your bank info.", "Click the link below for immediate access.",
        "Limited time only — call our toll-free hotline.", "100% satisfaction guaranteed."
    ]
    spam_signoffs = [
        "Prize Distribution Department", "Global Lottery Group", "Account Verification Team",
        "Pharma Online Direct", "Customer Rewards Center", "VIP Wealth Management"
    ]

    ham_intros = [
        "Hi team,", "Good morning,", "Hello everyone,", "Dear colleagues,",
        "Hi all,", "Following up on our earlier discussion,", "Hope you are doing well.",
        "Quick update regarding the project:", "Please note the following changes:"
    ]
    ham_actions = [
        "please find attached the quarterly budget review and financial statements",
        "we have scheduled the sprint review meeting for tomorrow at 2:00 PM in Conference Room B",
        "thank you for submitting the project milestone documentation on time",
        "the client approved our latest architectural proposal and contract terms",
        "please review the attached vendor invoice and forward approval by end of day",
        "our team completed the initial system testing with positive results",
        "let us sync on Monday morning to discuss the deployment schedule and deliverables",
        "here is the updated agenda for the upcoming stakeholder presentation",
        "the engineering workshop slides have been uploaded to the shared intranet drive",
        "can you provide feedback on the draft report before we distribute it to leadership"
    ]
    ham_urgency = [
        "Let me know if you have any questions or concerns.",
        "Regards and have a great weekend.",
        "Please confirm your attendance on the calendar invite.",
        "Looking forward to our discussion.",
        "Thanks again for all your hard work on this.",
        "Best regards,"
    ]
    ham_signoffs = [
        "Engineering Department", "Project Management Office", "Operations Team",
        "Finance and Accounting", "Academic Coordinator", "Product Engineering"
    ]

    records = []
    half = num_samples // 2

    # Generate Spam rows
    for _ in range(half):
        intro = random.choice(spam_intros)
        action = random.choice(spam_actions)
        urgency = random.choice(spam_urgency)
        signoff = random.choice(spam_signoffs)
        # Add random variations
        extra_keywords = random.sample([
            "click here", "free trial", "risk-free", "winner", "cash payout",
            "urgent", "credit score", "wire transfer", "bonus", "guaranteed"
        ], k=random.randint(1, 3))
        text = f"{intro} {action}. {urgency} Key terms: {', '.join(extra_keywords)}. Sincerely, {signoff}."
        records.append({"text": text, "label": "spam"})

    # Generate Ham rows
    for _ in range(half):
        intro = random.choice(ham_intros)
        action = random.choice(ham_actions)
        urgency = random.choice(ham_urgency)
        signoff = random.choice(ham_signoffs)
        extra_context = random.sample([
            "meeting notes", "attached spreadsheet", "quarterly metrics",
            "follow up", "team collaboration", "deliverables", "schedule"
        ], k=random.randint(1, 2))
        text = f"{intro} {action}. {urgency} Related to: {', '.join(extra_context)}. Regards, {signoff}."
        records.append({"text": text, "label": "ham"})

    random.shuffle(records)
    print("\n" + "="*70)
    print("WARNING: Using synthetic fallback dataset — real dataset unavailable.")
    print("="*70 + "\n")
    return pd.DataFrame(records)

def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame | None:
    """Normalizes various common spam dataset column naming conventions to (text, label)."""
    cols = {col.lower(): col for col in df.columns}
    
    # Common text column names
    text_col = None
    for cand in ["text", "v2", "message", "email", "content", "sms", "body"]:
        if cand in cols:
            text_col = cols[cand]
            break

    # Common label column names
    label_col = None
    for cand in ["label", "v1", "class", "target", "category", "spam"]:
        if cand in cols:
            label_col = cols[cand]
            break

    if text_col is None or label_col is None:
        if len(df.columns) >= 2:
            label_col, text_col = df.columns[0], df.columns[1]
        else:
            return None

    standardized = pd.DataFrame()
    standardized["text"] = df[text_col].astype(str)
    
    # Normalize labels: 1/0, spam/ham, etc.
    raw_labels = df[label_col].astype(str).str.lower().str.strip()
    
    def map_label(val):
        if val in ["1", "spam", "1.0", "true", "yes"]:
            return "spam"
        elif val in ["0", "ham", "0.0", "false", "no", "legit"]:
            return "ham"
        return "spam" if "spam" in val else "ham"

    standardized["label"] = raw_labels.apply(map_label)
    standardized = standardized.dropna().drop_duplicates(subset=["text"])
    standardized = standardized[standardized["text"].str.strip() != ""]
    return standardized

def load_data(force_refresh: bool = False) -> pd.DataFrame:
    """
    Main entry point: ensures data/raw/dataset.csv exists, executes fallback chain
    if necessary, and returns clean (text, label) DataFrame.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if DATASET_PATH.exists() and not force_refresh:
        df = pd.read_csv(DATASET_PATH)
        if "text" in df.columns and "label" in df.columns and len(df) > 100:
            print(f"[DataLoader] Loaded cached dataset from {DATASET_PATH} ({len(df)} rows)")
            return df

    # Fallback Tier 1: Kaggle
    df = try_load_kaggle()
    
    # Fallback Tier 2: GitHub Raw Mirror
    if df is None:
        df = try_load_github_mirror()
        
    # Fallback Tier 3: Synthetic Dataset
    if df is None:
        df = generate_synthetic_dataset(num_samples=2400)

    df.to_csv(DATASET_PATH, index=False)
    print(f"[DataLoader] Saved {len(df)} rows to {DATASET_PATH}")
    return df

if __name__ == "__main__":
    df = load_data()
    print("Class distribution:\n", df["label"].value_counts())
