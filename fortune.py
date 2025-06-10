import datetime
import random

FORTUNES = [
    "大吉: 素晴らしい一日になるでしょう。",
    "中吉: 良いことが起こりそうです。",
    "小吉: 平穏な一日になるでしょう。",
    "吉: 何か新しい発見があるかもしれません。",
    "凶: 無理せず慎重に過ごしましょう。",
]


def get_daily_fortune(date: datetime.date | None = None) -> str:
    """Return a deterministic fortune for the given date."""
    if date is None:
        date = datetime.date.today()
    random.seed(date.toordinal())
    return random.choice(FORTUNES)


def main() -> None:
    fortune = get_daily_fortune()
    print(f"本日の運勢: {fortune}")


if __name__ == "__main__":
    main()
