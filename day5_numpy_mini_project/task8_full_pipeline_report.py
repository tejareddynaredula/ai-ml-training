import numpy as np


N_STUDENTS = 200
N_SUBJECTS = 5
AT_RISK_THRESHOLD = 40
RANDOM_SEED = 42
REPORT_FILE = "day5_numpy_mini_project/student_analytics_report.txt"


def generate_dataset():
    np.random.seed(RANDOM_SEED)
    return np.random.randint(0, 101, size=(N_STUDENTS, N_SUBJECTS))


def compute_subject_stats(scores):
    return {
        "mean": np.mean(scores, axis=0),
        "median": np.median(scores, axis=0),
        "std": np.std(scores, axis=0),
        "min": np.min(scores, axis=0),
        "max": np.max(scores, axis=0),
    }


def rank_students(scores):
    totals = np.sum(scores, axis=1)
    return np.argsort(totals)[::-1]


def identify_at_risk_students(scores, threshold):
    averages = np.mean(scores, axis=1)
    return np.where(averages < threshold)[0]


def normalize_scores(scores):
    mean = np.mean(scores, axis=0)
    std = np.std(scores, axis=0)
    return (scores - mean) / std


def correlation_between_subjects(scores):
    return np.corrcoef(scores, rowvar=False)


def save_report(stats, ranking, at_risk, correlation, filepath):
    with open(filepath, "w") as file:
        file.write("STUDENT PERFORMANCE ANALYTICS REPORT\n")
        file.write("====================================\n\n")

        file.write(f"Students: {N_STUDENTS}\n")
        file.write(f"Subjects: {N_SUBJECTS}\n\n")

        file.write("SUBJECT STATISTICS\n")
        file.write("------------------\n")
        file.write(f"Mean: {stats['mean']}\n")
        file.write(f"Median: {stats['median']}\n")
        file.write(f"Standard Deviation: {stats['std']}\n")
        file.write(f"Minimum: {stats['min']}\n")
        file.write(f"Maximum: {stats['max']}\n\n")

        file.write("TOP 10 STUDENTS\n")
        file.write("---------------\n")
        file.write(f"Student indices: {ranking[:10]}\n\n")

        file.write("AT-RISK STUDENTS\n")
        file.write("----------------\n")
        file.write(f"Threshold: {AT_RISK_THRESHOLD}\n")
        file.write(f"Number of at-risk students: {len(at_risk)}\n")
        file.write(f"Student indices: {at_risk}\n\n")

        file.write("SUBJECT CORRELATION\n")
        file.write("-------------------\n")
        file.write(f"{np.round(correlation, 3)}\n")


def main():
    scores = generate_dataset()

    stats = compute_subject_stats(scores)
    ranking = rank_students(scores)
    at_risk = identify_at_risk_students(scores, AT_RISK_THRESHOLD)
    normalized = normalize_scores(scores)
    correlation = correlation_between_subjects(scores)

    save_report(
        stats,
        ranking,
        at_risk,
        correlation,
        REPORT_FILE,
    )

    print("Student Analytics Pipeline")
    print("==========================")
    print("Dataset shape:", scores.shape)
    print("At-risk students:", len(at_risk))
    print("Top student index:", ranking[0])
    print("Top student total:", np.sum(scores[ranking[0]]))
    print("Normalized mean:", np.round(np.mean(normalized, axis=0), 4))
    print("Report generated:", REPORT_FILE)


if __name__ == "__main__":
    main()