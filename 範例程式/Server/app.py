#架設 server
import os
import re
import time
import wave
import json
import openai
import pickle
import requests
from tdx import TDX
from gtts import gTTS # gtts-cli --all 查找語言及語系
from pytube import YouTube
from dotenv import load_dotenv
from pydub import AudioSegment
from googlesearch import search
import speech_recognition as sr
from langchain.tools import YouTubeSearchTool
from flask import Flask, request, send_file, abort

load_dotenv()
env_key = "OPENAI_API_KEY"
if env_key in os.environ and os.environ[env_key]:
    openai.api_key = os.environ[env_key]

TDX_ID='meebox-cc6ed12e-5254-47e3'
TDX_SECRET='b5bfb7cc-4b43-4f4f-97f8-faaf7705df9b'
tdx_client = TDX(TDX_ID,TDX_SECRET)

api_base_url1 = ('https://tdx.transportdata.tw/api/basic/v3/Rail/TRA/') # 台鐵
api_base_url2 = ('https://tdx.transportdata.tw/api/basic/v2/Rail/THSR/') # 高鐵
headers = {'user-agent':'Mozilla/5.0'}

def station_info():
    res = requests.get(f"{api_base_url1}Station?$select=stationName,stationID", headers=headers, )
    json_data = res.json()
    stations = {}
    for station in json_data['Stations']:
        station_name = station['StationName']['Zh_tw']
        station_id = station['StationID']
        stations[station_name] = station_id
    # print(stations)
    return stations

def high_station_info():
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
        headers=headers,)
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
            'train_no': train_no,
            'start_station': start_station,
            'departure_Time': departure_Time,
            'end_station': end_station,
            'arrive_Time': arrive_Time
        })
    # print(timetables[0])
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
    timetables1 = []
    timetables2 = []
    for timetable in json_data:
        train_no = timetable['DailyTrainInfo']['TrainNo']
        stop_times = timetable['OriginStopTime']
        start_station = stop_times['StationName']['Zh_tw']
        departure_Time = stop_times['DepartureTime']
        stop_times2 = timetable['DestinationStopTime']
        end_station = stop_times2['StationName']['Zh_tw']
        arrive_Time = stop_times2['ArrivalTime']
        timetables1.append({
            '起站': start_station,
            '出發': departure_Time,
            '到站': end_station,
            '抵達': arrive_Time
        })
        timetables2.append({
            '車次': train_no,
            '起站': start_station,
            '出發': departure_Time,
            '終站': end_station,
            '抵達': arrive_Time
        })
    # print(timetables[0])
    return timetables1,timetables2

# ==============儲存聊天歷史===============
def save_hist(hist):
    try:
        with open('hist.dat', 'wb') as f:
            pickle.dump(hist, f)
    except:
        # 歷史檔開啟錯誤
        print('無法寫入歷史檔')

# ==============載入聊天歷史===============
def load_hist(backtrace):
    try:
        with open('hist.dat', 'rb') as f:
            hist = pickle.load(f)
            return hist #['hist1'], db['sys_msg1'], db['hist2'], db['sys_msg2'], db['hist3'], db['sys_msg3']
    except:
        # 歷史檔不存在
        print('無法開啟歷史檔')
        return ['','']*backtrace


# ==============刪除檔案===============
def delete_file(file_path):
    try:
        os.remove(file_path)
    except Exception as e:
        pass

# ==============音樂網址===============
def get_music_url(music_name):
    music_url = tool.run(f'{music_name},1')
    music_url = eval(music_url)[0]
    return music_url

# ==============convert to wav===============
def convert_to_wav(input_file, output_file,format): # 讀取檔案名稱 輸出檔案名稱
    audio = AudioSegment.from_file(input_file, format=format) # 將 mp3 轉成 wav
    audio.export(output_file, format="wav") # wav 存檔

# ==============youtube to wav===============
def youtube_to_wav(url):
    yt = YouTube(url)
    print('download...',end='')
    file_name = os.path.join(app.config['UPLOAD_FOLDER'], 'music.mp4')
    yt.streams.filter().get_audio_only().download(filename=file_name)
    file_name_wav = os.path.join(app.config['UPLOAD_FOLDER'], 'music.wav')
    convert_to_wav(file_name, file_name_wav,"mp4")
    audio = AudioSegment.from_file(file_name_wav, format="wav")
    audio = audio.set_frame_rate(15000)
    audio = audio.set_channels(2)
    audio = audio.set_sample_width(2)
    audio.export(file_name_wav, format="wav")
    # change_sound = add_wav_volume(file_name_wav, -20)
    # change_sound.export(file_name_wav, format="wav")  # WAV 存檔
    print('ok')
    return 'ok'


# ==============wav檔音量增大===============
def add_wav_volume(filename, db):
    sound = AudioSegment.from_file(filename, "wav")
    try: # 嘗試增大音量
        # sound.dBFS 會試圖計算音訊數據的均方根（RMS），假如音訊數據的長度不是整數幀就會出錯。
        change_db = sound.dBFS+db
        change_dBFS = change_db - sound.dBFS
        return sound.apply_gain(change_dBFS)
    except Exception as e:
        print('音量增大失敗',e)
        return sound


# ==============pcm to wav===============
def pcm2wav(pcm_file, wav_file, channels=1, bits=16, sample_rate=8000):
    pcmf = open(pcm_file, 'rb')
    pcmdata = pcmf.read()
    pcmf.close()

    if bits % 8 != 0:
        raise ValueError("bits % 8 must == 0. now bits:" + str(bits))
    # 計算整數偵大小
    integer_frame_size = channels * (bits // 8)
    extra_bytes = len(pcmdata) % integer_frame_size
    if extra_bytes != 0:
        # 如果不是整數偵大小的倍數，則填充零偵至整數偵
        pcmdata += b'\x00' * (integer_frame_size - extra_bytes)

    wavfile = wave.open(wav_file, 'wb')
    wavfile.setnchannels(channels)
    wavfile.setsampwidth(bits // 8)
    wavfile.setframerate(sample_rate)
    wavfile.writeframes(pcmdata)
    wavfile.close()


# ==============語音轉文字===============
def convert_wav_to_text(wav_file):
    recognizer = sr.Recognizer()
    # 讀取 WAV 檔案
    with sr.AudioFile(wav_file) as source:
        try:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data, language="zh-TW")  # 使用Google語音識別，語言設置為繁體中文
            return text
        except sr.UnknownValueError:
            return "無法識別語音"
        except sr.RequestError as e:
            return f"發生錯誤：{e}"


# ==============text_to_speech===============
def text_to_speech(text, lang): # 文本 語言-語系
    tts = gTTS(text, lang=lang)  # 建立 gTTS
    # 將語音儲存為 mp3 檔案
    file_path = f'{path}/temp.mp3'  # mp3檔案路徑
    tts.save(file_path) # 儲存音檔
    output_file = f'{path}/uploads/temp.wav' # wav檔案路徑
    convert_to_wav(file_path, output_file,"mp3") # mp3 to wav
    # change_sound = add_wav_volume(output_file, -20)
    # change_sound.export(output_file, format="wav")  # WAV 存檔
    delete_file(file_path) # 刪除mp3
    return output_file

def chatgpt(system=None,messages=None,messages_dict=None):
    if messages_dict == None:
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


translate_language = '''
請取出以下敘述中出現的語言名稱並將其轉換成ISO 639-1的代號，
```
{}
```
請以下列 JSON 格式回答我, 除了 JSON 格式資料外,
不要加上額外資訊：
```
{{
    "name":"語言名稱"
    "language":"ISO 639-1的代號"
}}
```
如果不會, 請以下列 JSON 格式回答我：
```
{{
    "name":"抱歉，我不知道"
    "language":"zh-tw"
}}
'''

translate_something = '''
請判斷以下文句，篩出你覺得需要被翻譯的完整文句
```
{}
```
請以下列 JSON 格式回答我, 除了 JSON 格式資料外,
不要加上額外資訊：
```
{{
    "something":"需要被翻譯的文句"
}}
```
如果不會, 請以下列 JSON 格式回答我：
```
{{
    "something":"抱歉，我不知道"
}}
'''

translate_text = '''
請幫我翻譯以下文句成{}
```
{}
```
請以下列 JSON 格式回答我, 除了 JSON 格式資料外,
不要加上額外資訊：
```
{{
    "translate":"你翻譯的結果"
    "language":"ISO 639-1的代號"
}}
```
如果不會, 請以下列 JSON 格式回答我：
```
{{
    "translate":"抱歉，我不會翻"
    "language":"zh-tw"
}}
'''

function_choice = '''
請將以下文句選擇符合的代號,不要加上額外資訊
```
{}
```
需要開燈:t1,白色:t1,紅色燈:t2綠色燈:t3藍色燈:t4,黃色燈:t5,紫色燈:t6,藍綠色燈:t7,
循環燈:t8,彩色燈:t8,需要關燈:t9,音樂or歌曲or想聽:t10,翻譯:t11,火車:t12,其他:None
請以下列 JSON 格式回答我, 除了 JSON 格式資料外,
不要加上額外資訊：
```
{{
    "function_code":"你選擇的代號"
}}
```
如果不會, 請以下列 JSON 格式回答我：
```
{{
    "function_code":"None"
}}
'''

get_music_name = '''
請將以下文句中的歌曲名稱篩選出來,不要加上額外資訊
```
{}
```
請以下列 JSON 格式回答我, 除了 JSON 格式資料外,
不要加上額外資訊：
```
{{
    "music_name":"歌曲名稱"
}}
```
如果不會, 請以下列 JSON 格式回答我：
```
{{
    "music_name":"None"
}}
'''

station_name = '''
這裡提供高鐵車站代號表:```{}```
台鐵車站代號表:```{}```
請將以下文句中判斷台鐵或高鐵並取得起迄車站代號,不要加上額外資訊```{}```
請以下列 JSON 格式回答我, 除了 JSON 格式資料外,不要加上額外資訊：
```{{
    "train":"台鐵或高鐵"
    "station1":"起始站代號",
    "station2":"終點站代號"
    }}
```
如果不會, 請以下列 JSON 格式回答我：
```{{
    "train":"None"
    "station1":"None",
    "station2":"None"
    }}
'''

high_station_time = '''
這裡提供高鐵時刻表```{}```
請以下文句中選出最符合的出發時間,不要加上額外資訊```{}```
請以下列 JSON 格式回答我, 除了 JSON 格式資料外,不要加上額外資訊：
```{{
    "time1":"出發時間"
    }}
```
如果不會, 請以下列 JSON 格式回答我：
```{{
    "time1":"None"
    }}
'''

# 用來詢問是否需要搜尋才能回覆問題的樣板
# 要求 AI 以 JSON 格式回覆 Y/N 以及建議的搜尋關鍵字
template_google = '''
如果我想知道以下這件事, 請確認是否需要網路搜尋才做得到？

```
{}
```

如果你不知道, 請以下列 JSON 格式回答我, 除了 JSON 格式資料外,
不要加上額外資訊, 就算你知道答案, 也不要回覆：

```
{{
    "search":"Y",#你不知道就輸出Y
    "keyword":"你建議的搜尋關鍵字"
}}
```
如果不需要, 請以下列 JSON 格式回答我：

```
{{
    "search":"N",
    "keyword":""
}}
'''


# 取得主程式的目錄路徑
path = str(os.path.dirname(os.path.abspath(__file__)))
# 檢查是否存在 uploads 資料夾
uploads_directory = os.path.join(path, 'uploads')

if not os.path.exists(uploads_directory):
    # 如果 uploads 資料夾不存在，則建立它
    os.makedirs(uploads_directory)
    print("已建立 uploads 資料夾")
else:
    pass

light_list = {'t1':'白色燈','t2':'紅色燈','t3':'綠色燈','t4':'藍色燈','t5':'黃色燈','t6':'紫色燈','t7':'藍綠色燈','t8':'循環燈','t9':'關燈'}

tool = YouTubeSearchTool()
app = Flask(__name__) # 建立 app

UPLOAD_FOLDER = path+'/uploads'  # 上傳資料的資料夾
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 根目錄
@app.route('/')
def hello():
    return "Welcome to the audio server!"

 # 上傳文字目錄
@app.route('/upload_string', methods=['POST'])
def upload_string():
    try:
        # 接收從MicroPython設備發送的字串
        uploaded_data = request.data.decode('utf-8')
        data,lang = uploaded_data.split('$') # 字串中間有用$分隔，文本$語言
        print(data,lang)
        text_to_speech(data, lang) # 文字轉語音檔 /uploads/temp.wav
        return '接收成功'
    except Exception as e:
        print(uploaded_data)
        return '接收失敗: ' + str(e)

# 上傳PCM音檔
@app.route('/upload_audio', methods=['POST'])
def upload_audio():
    audio_data = request.data
    # 接收到的數據大小
    data_size = len(audio_data)
    #print(f'Received {data_size} bytes')
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'input.pcm')
    with open(file_path, 'ab') as audio_file:
        audio_file.write(audio_data)
    return '上傳成功'


# 處理音訊並與chatgpt溝通
@app.route('/fix_audio', methods=['GET'])
def fix_audio():
    # try:
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'input.pcm')
    wav_file = 'uploads/output.wav'
    pcm2wav(file_path, wav_file, channels=1, bits=16, sample_rate=8000) # PCM TO wav
    delete_file(file_path) # 刪除 PCM

    change_sound = add_wav_volume(wav_file, db=30) # 增加音量
    delete_file(wav_file)  # 刪除原始 WAV
    wav_file2 = "uploads/out.wav"
    change_sound.export(wav_file2, format="wav")  # WAV 存檔
    result = convert_wav_to_text(wav_file2) # 語音轉文字
    delete_file(wav_file2)  # 刪除音量增大 WAV
    # result = input('user: ')
    result = '龍井到大肚早上十點的火車'
    if result == '無法識別語音':
        print(result)
        result
        delete_file(f'{path}/uploads/temp.wav')
        text_to_speech(result, "zh-tw")  # 文字轉語音檔 /uploads/temp.wav
        return f'${result}$'
    else:
        print(f"USER：{result}")
        # AI判別是否啟用功能或一般對話
        reply = chatgpt('你是一個智能助手',function_choice.format(result))
        reply_json = json.loads(reply)
        reply = reply_json["function_code"]
        # print(reply)
        for i in range(12, 0,-1):
            if f't{i}' in reply:
                reply = f't{i}'
                # ==================================音樂撥放器======================================================
                if 't10' == reply:
                    music_reply = chatgpt('你是我的個人助理',get_music_name.format(result))
                    reply_json = json.loads(music_reply)
                    music_name = reply_json["music_name"]
                    if music_name == 'None':
                        music_name = '無法提取歌名'
                    else:
                        music_url = get_music_url(music_name)
                        text_to_speech(f'正在為您播放{music_name}', "zh-tw")  # 文字轉語音檔 /uploads/temp.wav
                        try:
                            youtube_to_wav(music_url)
                        except Exception as e:
                            reply='播放失敗'
                            print(e)
                            text_to_speech(reply, "zh-tw")  # 文字轉語音檔 /uploads/temp.wav
                        return f'{result}${reply}$正在為您播放{music_name}'
                # ========================================翻譯機======================================================
                elif 't11' == reply:
                    # 先確認要翻譯成什麼語言
                    language_reply = chatgpt('你是一個專業的翻譯員',translate_language.format(result))
                    reply_json = json.loads(language_reply)
                    language_name = reply_json["name"]
                    language = reply_json["language"]
                    # 取出此語言的名稱後，將它從文具中剃除(可以請AI做一次篩除)
                    # result = result.replace(language_name, '')

                    something_reply = chatgpt('你是一個專業的翻譯員',translate_something.format(result))
                    reply_json = json.loads(something_reply)
                    something = reply_json["something"]

                    anser_reply = chatgpt('你是一個專業的翻譯員，請將以下翻成{language_name}',
                                          translate_text.format(language_name, something))
                    reply_json = json.loads(anser_reply)
                    reply, language = reply_json["translate"], reply_json["language"]
                    print(f"Chatgpt：{reply}")
                    try:
                        text_to_speech(reply, language)  # 文字轉語音檔 /uploads/temp.wav
                    except:reply='翻譯錯誤'
                    return f'{result}${reply}$'
                # ========================================高鐵/台鐵時刻查詢======================================================
                elif 't12' == reply:
                    stations = station_info()
                    high_stations = high_station_info()
                    messages = station_name.format(high_stations, stations, result)
                    station_reply = chatgpt('', messages)
                    reply_json = json.loads(station_reply)
                    train = reply_json["train"]
                    start_station = reply_json["station1"]
                    end_station = reply_json["station2"]
                    # ========================================判斷高鐵或台鐵======================================================
                    if train == '高鐵':
                        timetables1, timetables2 = high_stations_time(start_station, end_station)
                        messages = high_station_time.format(timetables1, result)
                        time_reply = chatgpt('', messages)
                        reply_json = json.loads(time_reply)
                        time1 = reply_json["time1"]
                        m_i = []
                        for i in timetables2:
                            if i['出發'] == time1:
                                m_i += [i]

                        messages = f'{str(m_i)}根據此資料請簡述回答以下問題:{result}'
                        reply = chatgpt('你是我的個人助理，並且只能用繁體中文回答', messages)
                        print(f"Chatgpt：{reply}")
                        text_to_speech(reply, "zh-tw")  # 文字轉語音檔 /uploads/temp.wav
                        return f'{result}${reply}$'
                    else: # 台鐵
                        timetables = stations_time(start_station, end_station)
                        messages = f'{timetables}請詳細回答: {result}'
                        reply = chatgpt('你是我的個人助理，並且只用繁體中文回答', messages)
                        print(f"Chatgpt：{reply}")
                        text_to_speech(reply, "zh-tw")  # 文字轉語音檔 /uploads/temp.wav
                        return f'{result}${reply}$'
                # ========================================控制led燈======================================================
                else:
                    light = light_list[reply]

                    if reply == 't9': reply_light = f'已為您{light}'
                    elif reply == 't8':reply_light = f'開啟{light}'
                    else:reply_light = f'已為您開啟{light}'
                    text_to_speech(reply_light, 'zh-tw')  # 文字轉語音檔 /uploads/temp.wav
                    print(f"Chatgpt：{reply_light}")
                    return f'{result}${reply}${reply_light}'
                break
        # ========================================聊天模式======================================================
        else :

            backtrace = 3
            hist = load_hist(backtrace)
            hist_len = backtrace * 2
            message = []
            # -------------------------------逐一加上對話紀錄---------------------------
            for i in range(0, hist_len, 2):
                if hist[i] == '':
                    continue
                message.append({'role': 'user', 'content': hist[i]})
                message.append({'role': 'assistant', 'content': hist[i + 1]})
            # ------------------------------------------------------------------------
            message1 = message.copy()
            message1.append({'role': 'user', 'content': template_google.format(result)}) # 餵給googlesearch樣板

            go_reply = chatgpt(None,None,message1)
            reply_json = json.loads(go_reply)
            search_Y_N = reply_json["search"]
            keyword = reply_json["keyword"]
            print(keyword)
            if search_Y_N == 'Y':
                content = "最新資訊：\n"
                i_c = 1
                for item in search(keyword, advanced=True, num_results=5):
                    content += f"標題：{item.title}\n"
                    content += f"摘要：{item.description}\n\n"
                content += "只需單純用繁體中文回答以下問題，如果遇到科學單位請以台灣常用的為準(°C、kg、m/s等等)，不要加上額外資訊：\n"
                result_2 = content + result
            else:# 中和天氣如何
                result_2 = result
            message.append({'role': 'user', 'content': result_2})  # 最後加上本次使用者訊息
            print(message)
            reply = chatgpt(None,None,message)  # chatgpt
            user = result
            while len(hist) >= hist_len:
                hist.pop(1)
                hist.pop(0)
            hist += [user, reply]  # 紀錄對話
            save_hist(hist)  # 儲存對話
            text_to_speech(reply, "zh-tw")  # 文字轉語音檔 /uploads/temp.wav
        print(f"Chatgpt：{reply}")
        return f'{result}${reply}$'
    # except Exception as e:
    #     return 'Error : ' + str(e)




# 下載音檔的路徑
@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.exists(file_path): # 確認有回覆的音檔
        return send_file(file_path, as_attachment=True)
    else:
        return abort(404)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
