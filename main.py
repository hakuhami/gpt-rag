from scripts.run_analysis import run_analysis

def main():
    config_path = 'config/config.yml'
    input_text = """
    当社は、2030年までにCO2排出量を2015年比で50%削減することを目指しています。
    この目標達成に向けて、再生可能エネルギーの導入を進めており、2022年度には
    全電力使用量の30%を再生可能エネルギーで賄いました。
    """
    run_analysis(config_path, input_text)

if __name__ == "__main__":
    main()