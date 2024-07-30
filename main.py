# This branch is used to analyze the English data.

from scripts.run_analysis import run_analysis

def main():
    config_path = 'config/config.yml'
    run_analysis(config_path)

if __name__ == "__main__":
    main()