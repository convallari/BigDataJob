#%%
import re
from time import sleep

import requests
import json
import time
from bs4 import BeautifulSoup
import pandas as pd
import time

headers = {
    'Accept':'application/json, text/plain, */*',
    'Accept-Encoding':'gzip, deflate, br, zstd',
    'Accept-Language':'zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6',
    'Connection':'keep-alive',
    'Host':'yiqifu.baidu.com',
    'Referer':'https://yiqifu.baidu.com/g/aqc/joblist?q=python',
    'Sec-Ch-Ua':'Microsoft Edge";v="131", "Chromium";v="131", "Not_A Brand";v="24',
    'Sec-Ch-Ua-Mobile':'?0',
    'Sec-Ch-Ua-Platform':'"Windows"',
    'Sec-Fetch-Dest':'empty',
    'Sec-Fetch-Mode':'cors',
    'Sec-Fetch-Site':'same-origin',
    'X-Requested-With':'XMLHttpRequest',
    'Cookie':'BAIDUID=1A8C7374679D8573D13B08550BA73F87:FG=1; BAIDUID_BFESS=1A8C7374679D8573D13B08550BA73F87:FG=1; BIDUPSID=1A8C7374679D8573D13B08550BA73F87; PSTM=1725458716; H_PS_PSSID=60279_60360_60630_60665_60678_60684_60700_60726; ZFY=kYgTDpsmpnLh988swH4vJpmJP5heT2wkWvhRfkVqtSY:C; __bid_n=191df09a4ae8768b87fe4b; BDUSS=JhM0NrMXgzb0Z6RXB6NE9mTUN4ZXNEMkdlc0RpeWVEQ09-eTNldVM5bThsUWhuSUFBQUFBJCQAAAAAAAAAAAEAAAAmVz9eAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAALwI4Wa8COFmWW; BDUSS_BFESS=JhM0NrMXgzb0Z6RXB6NE9mTUN4ZXNEMkdlc0RpeWVEQ09-eTNldVM5bThsUWhuSUFBQUFBJCQAAAAAAAAAAAEAAAAmVz9eAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAALwI4Wa8COFmWW; jsdk-uuid=d9cff227-1fc5-45ea-a397-26a691d50e0d; RT="z=1&dm=baidu.com&si=519556ff-4e7f-4b92-a245-c2838e0ab0da&ss=m3y5y83z&sl=y&tt=sc8&bcn=https%3A%2F%2Ffclog.baidu.com%2Flog%2Fweirwood%3Ftype%3Dperf&ld=17cic&nu=9y8m6cy&cl=527k&ul=18lvr&hd=18lvx"; clue_site=pc; log_guid=7ac8fe20d6c76c027b60abd76feab900; log_first_time=1732876085548; Hm_lvt_37e1bd75d9c0b74f7b4a8ba07566c281=1732876086; HMACCOUNT=43B3FC0128F7977F; Hm_lpvt_37e1bd75d9c0b74f7b4a8ba07566c281=1732876939; log_last_time=1732877721240',
    'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36 Edg/131.0.0.0'
}

url = 'https://yiqifu.baidu.com/g/aqc/joblist/getDataAjax?'
district = 310000 #上海


def get_data(page):
    params = f'q=%E5%A4%A7%E6%95%B0%E6%8D%AE&page={page}&pagesize=20&district={district}&salaryrange='
    r = requests.get(url,headers=headers,params=params)
    r_js = json.loads(r.text)
    r_list = r_js['data']['list']
    return r_list


def process_job(data):
    job_data = {}   #每个职位信息用字典存储
    job_data['城市'] = data['city']
    job_data['公司名称'] = data['company']
    job_data['学历要求'] = data['edu']
    job_data['工作经验'] = data['exp']
    job_data['招聘岗位'] = data['jobName'].replace('<em>', '').replace('</em>', '')
    job_data['薪资待遇'] = data['salary']
    bid = data['bid']
    jobId = data['jobId']
    job_url = f'https://yiqifu.baidu.com/g/aqc/jobDetail?bid={bid}&jobId={jobId}&from=ps&fr=job_ald&rq=pos'
    job_data['职位描述'] = get_job_detail(job_url)
    job_data['职位描述'] = clean_text(job_data['职位描述'])
    print(f'已获取{job_data}')
    return job_data

def clean_text(text):
    # 删除所有非打印字符
    return re.sub(r'\x0b', '', text)

def get_job_detail(job_url):
    res = requests.get(job_url, headers=headers)
    bs = BeautifulSoup(res.text, "html.parser")
    scripts = bs.find_all("script")
    text = ""
    for script in scripts:
        if "window.pageData" in script.text:
            text = script.text
    start = text.find("window.pageData = ") + len("window.pageData = ")
    end = text.find(" || {}")
    job_des = text[start:end]
    if job_des:
        data = json.loads(job_des)
        if data and "desc" in data:
            time.sleep(1)
            return data["desc"].replace("<br />", "").replace("</p>", "").replace("<p>", "").replace("&nbsp;", "")
    return ""

def while_data(total_page):
    all_data = [] # 列表用于存放所有的职位信息
    for i in range(1, total_page+1):
        try:
            data = get_data(i)
            time.sleep(1)
            # 如果有获取到数据则进行处理
            if data:
                for item in data:
                    job = process_job(item)
                    all_data.append(job)
        except:
            sleep(10)
            print('出现错误，等待10秒')
            data = get_data(i)

    return all_data

total_data = while_data(20)
df = pd.DataFrame(total_data)
df.to_excel('job_bigData.xlsx',index=False)
