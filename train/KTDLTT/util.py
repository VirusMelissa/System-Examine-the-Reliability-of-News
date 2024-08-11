from DanTriCrawlingData import get_dantri_news_from_url
from EmdepCrawlingData import get_emdep_news_from_url
from GenkCrawlingData import get_genk_news_from_url
from Kenh14CrawlingData import get_kenh14_news_from_url
from LaoDongCrawlingData import get_laodong_news_from_url
from NhanDanCrawlingData import get_nhandan_news_from_url
from SaiGon24HCrawlingData import get_saigon24_news_from_url
from SohaCrawlingData import get_soha_news_from_url
from ThanhnienCrawlingData import get_thanhnien_news_from_url
from TuoiTreCrawlingData import get_tuoitre_news_from_url
from VNexpressCrawlingData import get_news_from_url
from VietNamNetCrawlingData import get_vietnamnet_news_from_url


def switchFN(lang, res):
    if lang == "dantri":
        return get_dantri_news_from_url(res, 1)
    elif lang == "nhandan":
        return get_nhandan_news_from_url(res, 1)
    elif lang == "laodong":
        return get_laodong_news_from_url(res, 1)
    elif lang == "tuoitre":
        return get_tuoitre_news_from_url(res, 1)
    elif lang == "thanhnien":
        return get_thanhnien_news_from_url(res, 1)
    elif lang == "vietnamnet":
        return get_vietnamnet_news_from_url(res, 0)
    elif lang == 'vnexpress':
        return get_news_from_url(res,1)
    elif lang == 'genk':
        return get_genk_news_from_url(res, 0)
    elif lang == 'kenh14':
        return get_kenh14_news_from_url(res, 0)
    elif lang == 'soha':
        return get_soha_news_from_url(res, 0)
    elif lang == 'emdep':
        return get_emdep_news_from_url(res, 0)
    elif lang == 'saigon24':
        return get_saigon24_news_from_url(res, 0)
    else: 
        return None