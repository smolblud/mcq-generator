import os
import re
import pandas as pd

# Folder containing all your TXT files
DATA_FOLDER = r"mcq-generator/data"

# Regex to detect headings (Chapters, Units, Lessons)
HEADING_PATTERNS = [
    r"chapter\s*\d+",
    r"unit\s*\d+",
    r"lesson\s*\d+",
    r"[A-Z][a-z]+(\s[A-Z][a-z]+)*\s*$"   # fallback: Capitalized line
]


def is_heading(line):
    """Check if a line looks like a topic heading."""
    line = line.strip()
    if len(line) < 3:
        return False

    for pattern in HEADING_PATTERNS:
        if re.search(pattern, line, re.IGNORECASE):
            return True

    return False


def extract_from_file(filepath):
    """Extract topics & subtopics from a single text file."""
    topics = []
    current_heading = None

    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()

            # Detect heading
            if is_heading(line):
                current_heading = line
                continue

            # Normal content becomes subtopic
            if current_heading and len(line) > 5:
                topics.append([os.path.basename(filepath), current_heading, line])

    return topics


def main():
    all_topics = []

    for root, dirs, files in os.walk(DATA_FOLDER):
        for file in files:
            if file.endswith(".txt"):
                path = os.path.join(root, file)
                print(f"Extracting: {path}")
                topics = extract_from_file(path)
                all_topics.extend(topics)

    # Convert to a blueprint table
    df = pd.DataFrame(all_topics, columns=["File", "Topic", "Subtopic"])

    # Add estimated MCQs
    df["Estimated_MCQs"] = df["Subtopic"].apply(lambda x: 5)

    # Save output
    df.to_csv("final_topic_blueprint.csv", index=False)
    print("Saved: final_topic_blueprint.csv")


if __name__ == "__main__":
    main()
