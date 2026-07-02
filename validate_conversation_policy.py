"""End-to-end behavioral checks for the conversational risk policy."""

from backend.main import AnalysisRequest, analyze


MODELS = ("lightgbm", "logistic", "svm", "mental_roberta")
SCENARIOS = {
    "daily_life": [
        "Work was tiring today, but overall I am okay.",
        "I cooked dinner and watched a movie.",
        "Tomorrow I plan to meet a friend.",
    ],
    "temporary_low_mood": [
        "Today was difficult and I feel exhausted.",
        "I have been sad and unmotivated this week.",
        "I talked to my family and feel a little better.",
        "I went outside and enjoyed some fresh air.",
        "I feel calmer and hopeful about tomorrow.",
    ],
    "crisis_and_recovery": [
        "I have been overwhelmed and unable to sleep.",
        "I want to end my life tonight and cannot continue anymore.",
        "I am safe now and called my sister for help.",
        "We are together and making a plan to get professional support.",
        "I ate dinner, feel calmer, and will not be alone tonight.",
        "I feel supported and hopeful about getting help tomorrow.",
    ],
}


def run(model: str, messages: list[str]) -> list[dict]:
    history = []
    results = []
    for message in messages:
        result = analyze(AnalysisRequest(
            text=message, history=history, language="en", model=model
        ))
        results.append(result)
        history.append(message)
    return results


def main() -> None:
    for model in MODELS:
        daily = run(model, SCENARIOS["daily_life"])
        assert all(item["label"] == "Normal" for item in daily), (model, daily)
        assert not any(item["immediateRisk"] or item["recentRisk"] for item in daily)

        low_mood = run(model, SCENARIOS["temporary_low_mood"])
        assert low_mood[-1]["label"] == "Normal", (model, low_mood[-1])
        assert not low_mood[-1]["immediateRisk"] and not low_mood[-1]["recentRisk"]

        recovery = run(model, SCENARIOS["crisis_and_recovery"])
        assert recovery[1]["immediateRisk"], (model, "crisis not detected")
        assert recovery[2]["recentRisk"] and recovery[3]["recentRisk"], model
        assert not recovery[4]["recentRisk"] and not recovery[5]["recentRisk"], model
        assert recovery[-1]["label"] != "Suicidal", (model, recovery[-1])
        print(model, "OK", " -> ".join(item["label"] for item in recovery))


if __name__ == "__main__":
    main()
