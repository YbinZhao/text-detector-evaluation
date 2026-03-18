export HF_ENDPOINT="https://hf-mirror.com"
export HF_HOME="D:/cache"
# python fastdetect.py --input_json arxiv_4000.json --output_json arxiv_fastgpt.json
# python fastdetect.py --input_json squad_4000.json --output_json squad_fastgpt.json
# python fastdetect.py --input_json writing_4000.json --output_json writing_fastgpt.json
# python fastdetect.py --input_json xsum_4000.json --output_json xsum_fastgpt.json
# python fastdetect.py --input_json yelp_4000.json --output_json yelp_fastgpt.json

# python binoculars.py --input_json arxiv_4000.json --output_json arxiv_binoculars.json
# python binoculars.py --input_json squad_4000.json --output_json squad_binoculars.json
# python binoculars.py --input_json writing_4000.json --output_json writing_binoculars.json
# python binoculars.py --input_json xsum_4000.json --output_json xsum_binoculars.json
# python binoculars.py --input_json yelp_4000.json --output_json yelp_binoculars.json

# python statistic.py --method entropy --threshold 2.9612514972686768 --direction gt --input_json arxiv_4000.json --output_json arxiv_entropy.json
# python statistic.py --method entropy --threshold 2.710803747177124 --direction gt --input_json squad_4000.json --output_json squad_entropy.json
# python statistic.py --method entropy --threshold 3.057915325164795 --direction gt --input_json writing_4000.json --output_json writing_entropy.json
# python statistic.py --method entropy --threshold 2.846575975418091 --direction gt --input_json xsum_4000.json --output_json xsum_entropy.json
# python statistic.py --method entropy --threshold 3.786832809448242 --direction gt --input_json yelp_4000.json --output_json yelp_entropy.json

# python statistic.py --method likelihood --threshold -2.239182949066162 --direction lt --input_json arxiv_4000.json --output_json arxiv_likelihood.json
# python statistic.py --method likelihood --threshold -2.5923547744750977 --direction lt --input_json squad_4000.json --output_json squad_likelihood.json
# python statistic.py --method likelihood --threshold -2.786428213119507 --direction lt --input_json writing_4000.json --output_json writing_likelihood.json
# python statistic.py --method likelihood --threshold -2.4663515090942383 --direction lt --input_json xsum_4000.json --output_json xsum_likelihood.json
# python statistic.py --method likelihood --threshold -2.5714352130889893 --direction lt --input_json yelp_4000.json --output_json yelp_likelihood.json

# python statistic.py --method logrank --threshold -1.119217038154602 --direction lt --input_json arxiv_4000.json --output_json arxiv_logrank.json
# python statistic.py --method logrank --threshold -1.2706235647201538 --direction lt --input_json squad_4000.json --output_json squad_logrank.json
# python statistic.py --method logrank --threshold -1.3878720998764038 --direction lt --input_json writing_4000.json --output_json writing_logrank.json
# python statistic.py --method logrank --threshold -1.2076376676559448 --direction lt --input_json xsum_4000.json --output_json xsum_logrank.json
# python statistic.py --method logrank --threshold -1.3275736570358276 --direction lt --input_json yelp_4000.json --output_json yelp_logrank.json

# python statistic.py --method rank --threshold -52.50980758666992 --direction lt --input_json arxiv_4000.json --output_json arxiv_rank.json
# python statistic.py --method rank --threshold -28.537687301635742 --direction lt --input_json squad_4000.json --output_json squad_rank.json
# python statistic.py --method rank --threshold -50.24623107910156 --direction lt --input_json writing_4000.json --output_json writing_rank.json
# python statistic.py --method rank --threshold -38.743717193603516 --direction lt --input_json xsum_4000.json --output_json xsum_rank.json
# python statistic.py --method rank --threshold -39.832881927490234 --direction lt --input_json yelp_4000.json --output_json yelp_rank.json

# python detectgpt.py --input_json arxiv_4000.json --output_json arxiv_detect.json
# python detectgpt.py --input_json squad_4000.json --output_json squad_detect.json
# python detectgpt.py --input_json writing_4000.json --output_json writing_detect.json
# python detectgpt.py --input_json xsum_4000.json --output_json xsum_detect.json
# python detectgpt.py --input_json yelp_4000.json --output_json yelp_detect.json


# python roberta_base.py --test_data_path D:/py/irt-benchmark/Benchmark/domain/arxiv_test_only.json
# python roberta_base.py --test_data_path D:/py/irt-benchmark/Benchmark/domain/squad_test_only.json
# python roberta_base.py --test_data_path D:/py/irt-benchmark/Benchmark/domain/writing_prompt_test_only.json
# python roberta_base.py --test_data_path D:/py/irt-benchmark/Benchmark/domain/xsum_test_only.json
# python roberta_base.py --test_data_path D:/py/irt-benchmark/Benchmark/domain/yelp_reviews_test_only.json

python roberta_large.py --test_data_path D:/py/irt-benchmark/Benchmark/domain/arxiv_test_only.json  
# python roberta_large.py --test_data_path D:/py/irt-benchmark/Benchmark/domain/squad_test_only.json 
# python roberta_large.py --test_data_path D:/py/irt-benchmark/Benchmark/domain/writing_prompt_test_only.json 
# python roberta_large.py --test_data_path D:/py/irt-benchmark/Benchmark/domain/xsum_test_only.json 
# python roberta_large.py --test_data_path D:/py/irt-benchmark/Benchmark/domain/yelp_reviews_test_only.json

# python chatgpt-detector-roberta.py --test_data_path D:/py/irt-benchmark/Benchmark/domain/arxiv_test_only.json 
# python chatgpt-detector-roberta.py --test_data_path D:/py/irt-benchmark/Benchmark/domain/squad_test_only.json 
# python chatgpt-detector-roberta.py --test_data_path D:/py/irt-benchmark/Benchmark/domain/writing_prompt_test_only.json 
# python chatgpt-detector-roberta.py --test_data_path D:/py/irt-benchmark/Benchmark/domain/xsum_test_only.json 
# python chatgpt-detector-roberta.py --test_data_path D:/py/irt-benchmark/Benchmark/domain/yelp_reviews_test_only.json

