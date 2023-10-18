import os
import json
import openai
import requests
from tdx import TDX
from dotenv import load_dotenv
import time
api_base_url1 = 'https://tdx.transportdata.tw/api/basic/v3/Rail/TRA/'  # 台鐵
api_base_url2 = 'https://tdx.transportdata.tw/api/basic/v2/Rail/THSR/'  # 高鐵
headers = {'user-agent':'Mozilla/5.0'}

load_dotenv()
env_key = "OPENAI_API_KEY"
if env_key in os.environ and os.environ[env_key]:
    openai.api_key = os.environ[env_key]
    TDX_ID = os.getenv('TDX_ID'),
    TDX_SECRET = os.getenv('TDX_SECRET')
tdx_client = TDX(TDX_ID,TDX_SECRET) # 讀取 api ID 與 密碼

station_name = '''
請將以下文句中判斷台鐵或高鐵並取得起迄車站代號,不要加上額外資訊```{}```
請以下列 JSON 格式回答我, 除了 JSON 格式資料外,不要加上額外資訊：
```{{
    "train":"台鐵或高鐵",
    "station1":"起始站名稱",
    "station2":"終點站名稱"
    }}
```
如果不會, 請以下列 JSON 格式回答我：
```{{
    "train":"None",
    "station1":"None",
    "station2":"None"
    }}
'''

high_station_time = '''
這裡提供整天高鐵時刻表的出發時間```{}```
請回答以下問題```{}```，並給出最符合的時間及前後共三個時間
請以下列 JSON 格式回答我, 除了 JSON 格式資料外,不要加上額外資訊：
```{{
    "time1":"前一個時間",
    "time2":"出發時間",
    "time3":"後一個時間",
    }}
```
如果不會, 請以下列 JSON 格式回答我：
```{{
    "time1":"None",
    "time2":"None"
    }}
'''

def station_info(trains):
    """
    取得台鐵或高鐵的所有停靠站
    Args:
        trains:1(台鐵) or 2(高鐵)
    Returns:
        stations
    """
    if trains == 1 :
        json_data = tdx_client.get_json(f"{api_base_url1}Station?$select=stationName,stationID")
        json_data = json_data['Stations']
    else:
        json_data = tdx_client.get_json(f"{api_base_url2}Station?$select=stationName,stationID")
    stations = {}
    for station in json_data:
        station_name = station['StationName']['Zh_tw']
        station_id = station['StationID']
        stations[station_name] = station_id
    return stations

def stations_time(start_station,end_station):
    now = time.localtime()
    date = f"{now.tm_year}-{now.tm_mon:02d}-{now.tm_mday:02d}"
    res = stations = requests.get(
        f"{api_base_url1}DailyTrainTimetable/OD/"
        f"{start_station}"
        "/to/"
        f"{end_station}"
        "/"
        f"{date}",
        headers=headers)
    json_data = res.json()
    timetables = []
    for timetable in json_data['TrainTimetables']:
        train_no = timetable['TrainInfo']['TrainNo']
        stop_times = timetable['StopTimes']
        start_station = stop_times[0]['StationName']['Zh_tw']
        departure_Time = stop_times[0]['DepartureTime']
        end_station = stop_times[1]['StationName']['Zh_tw']
        arrive_Time = stop_times[1]['ArrivalTime']
        timetables.append({
            "車次": train_no,
            "出發": departure_Time,
            "抵達": arrive_Time
        })
    # print(timetables[0])
    timetables = sorted(
        timetables,
        key=lambda timetable: timetable["出發"]
    )
    return timetables


def high_stations_time(start_station,end_station):
    now = time.localtime()
    date = f"{now.tm_year}-{now.tm_mon:02d}-{now.tm_mday:02d}"
    json_data = tdx_client.get_json(f"{api_base_url2}DailyTimetable/OD/"
        f"{start_station}"
        "/to/"
        f"{end_station}"
        "/"
        f"{date}")
    timetables = []
    timetables_full = []
    for timetable in json_data:
        train_no = timetable['DailyTrainInfo']['TrainNo']
        stop_times = timetable['OriginStopTime']
        start_station = stop_times['StationName']['Zh_tw']
        departure_Time = stop_times['DepartureTime']
        stop_times2 = timetable['DestinationStopTime']
        end_station = stop_times2['StationName']['Zh_tw']
        arrive_Time = stop_times2['ArrivalTime']
        timetables.append({
            "出發": departure_Time,
        })
        timetables_full.append({
            "車次": train_no,
            "出發": departure_Time,
            "抵達": arrive_Time
        })
    timetables = sorted(
        timetables,
        key=lambda timetable: timetable["出發"]
    )
    return timetables,timetables_full

def replace_common_chars(name: str):
    return name.replace('台', '臺')


def get_station_id(stations,station_flag,station_name):
    if station_flag == 1:
        station_name = replace_common_chars(station_name)
    else: pass
    res = stations.get(station_name, '0000')
    return res

def chatgpt(system=None,messages=None,messages_dict=None):
    """
    Args:
        system: 設定系統訊息
        messages: user的訊息
        messages_dict: 傳給chatgpt的dict
    Returns: reply--chatgpt的回答
    """
    if messages_dict == None: # 如果沒有傳入massages_dict，直接把system與message放入
        messages_dict = [{'role': 'system', 'content': system,
                          'role': 'user', 'content': messages}]
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages_dict,
        )
        return response['choices'][0]['message']['content']
    except openai.OpenAIError as err:
        reply = f"發生 {err.error.type} 錯誤\n{err.error.message}"
        return reply

def find_best_train(result):
    normal_stations = station_info(1)
    high_stations = station_info(2)
    messages = station_name.format(result)  # 帶入樣板取得 車種、起點、終點

    station_reply = chatgpt('', messages)
    reply_json = json.loads(station_reply)
    train = reply_json["train"]  # 台鐵或高鐵
    if train == '高鐵':
        station_flag = 0
        stations = high_stations
        start_station =reply_json["station1"]  # 起點站
        end_station =reply_json["station2"]  # 終點站
        start_id = get_station_id(stations, station_flag, start_station)
        end_id = get_station_id(stations, station_flag, end_station)
        timetables,timetables_full = high_stations_time(start_id, end_id)
        messages = high_station_time.format(timetables, result)
        reply = chatgpt('', messages)
        reply_json = json.loads(reply)
        time1 = reply_json['time1']
        time2 = reply_json['time2']
        time3 = reply_json['time3']
        if time1 == None:
            reply = '未找到最合適的車次'
        else:
            choice_num = []
            for item in timetables_full:
                if item['出發'] == time1 or item['出發'] == time2 or item['出發'] == time3:
                    choice_num.append(item)
            reply = chatgpt('', f'即時的高鐵時刻表:{choice_num}，請詳細回答以下問題，並給出三個車次作為選擇 {result}')

    elif train == '台鐵':
        station_flag = 1
        stations = normal_stations
        start_id = get_station_id(stations, station_flag, reply_json["station1"])  # 起點站
        end_id = get_station_id(stations, station_flag, reply_json["station2"])  # 終點站
        timetables = stations_time(start_id, end_id)
        messages = f'車次時刻表:{timetables}請根據時刻表回答最接近以下時間:{result}'
        station_reply = chatgpt('你是我的個人助理，並且只能用繁體中文回答', messages)
        reply = station_reply
    else: reply = '請再次查詢'
    return reply

# result = '台北到台中早上十一點出發的高鐵'
# print(find_best_train(result))