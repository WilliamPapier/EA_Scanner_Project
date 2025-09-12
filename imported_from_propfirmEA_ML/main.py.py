from src.scanner.market_scanner import run_scanner

def main():
    print("✅ Trading Bot is running correctly!")
    print("Next step: connecting scanner...")

    result = run_scanner()
    print("Scanner result:", result)

if __name__ == "__main__":
    main()
