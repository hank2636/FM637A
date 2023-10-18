'''All mode'''
#架設 server
import os
import re
import time
import wave
import json
import openai
import pickle
import requests
import Train_Module
from gtts import gTTS # gtts-cli --all 查找語言及語系
from pytube import YouTube
from dotenv import load_dotenv
from pydub import AudioSegment
from googlesearch import search #upm package(googlesearch-python)
import speech_recognition as sr
from langchain.tools import YouTubeSearchTool
from flask import Flask, request, send_file, abort


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
    audio = audio.set_frame_rate(20000)
    audio.export(output_file, format="wav") # wav 存檔

# ==============youtube to wav===============
def youtube_to_wav(url):
    yt = YouTube(url)
    print('download---',end='')
    file_name = os.path.join(app.config['UPLOAD_FOLDER'], 'music.mp4')
    yt.streams.filter().get_audio_only().download(filename=file_name)
    file_name_wav = os.path.join(app.config['UPLOAD_FOLDER'], 'music.wav')
    convert_to_wav(file_name, file_name_wav,"mp4")
    audio = AudioSegment.from_file(file_name_wav, format="wav")
    # audio = audio.set_frame_rate(24000)
    audio = audio.set_channels(1)
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

def speech_to_text():
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'input.pcm')
    wav_file = 'uploads/output.wav'
    pcm2wav(file_path, wav_file, channels=1, bits=16, sample_rate=8000)  # PCM TO wav
    delete_file(file_path)  # 刪除 PCM 檔
    change_sound = add_wav_volume(wav_file, db=30)  # 增加音量
    delete_file(wav_file)  # 刪除原始 WAV 檔
    wav_file2 = "uploads/out.wav"
    change_sound.export(wav_file2, format="wav")  # WAV 存檔
    result = convert_wav_to_text(wav_file2)  # 語音轉文字
    delete_file(wav_file2)  # 刪除音量增大 WAV 檔
    return result

# ==============text_to_speech===============
def text_to_speech(text, lang_param): # 文本 語言-語系
    """
    Args:
        text: 傳入的文字
        lang: 語言-語系 e.g. "zh-TW"
    Returns: output_file--wav檔案路徑,測試時方便觀察
    """
    tts = gTTS(text, lang= lang_param)  # 建立 gTTS
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

def catch_destination(result):
    stations = station_info(1)  # 取得台鐵站名代號
    high_stations = station_info(0)  # 取得高鐵站名代號
    messages = station_name.format(high_stations, stations, result)  # 帶入樣板取得 車種、起點、終點
    station_reply = chatgpt('', messages)
    reply_json = json.loads(station_reply)
    train = reply_json["train"]  # 台鐵或高鐵
    start_station = reply_json["station1"]  # 起點站
    end_station = reply_json["station2"]  # 終點站
    return train,start_station,end_station

def catch_train_number(start_station, end_station, result):
    timetables1, timetables2 = high_stations_time(start_station, end_station)  # 取得時刻表(純時間)、時刻表(含車次)
    messages = high_station_time.format(timetables1, result)  # 找出最接近的時間(樣板)
    time_reply = chatgpt('', messages)
    reply_json = json.loads(time_reply)
    time1 = reply_json["time1"]
    for i in timetables2:
        if i['出發'] == time1:  # 找到相對應的出發時間
            m_i = [i]  # 有對應到時間的車次
            return m_i

def save_chat(user, reply):
    backtrace = 3
    hist = load_hist(backtrace)
    hist_len = backtrace * 2
    while len(hist) >= hist_len:
        hist.pop(1)
        hist.pop(0)
    hist += [user, reply]  # 紀錄對話
    save_hist(hist)  # 儲存對話

 # 音樂撥放器
def player_machine(result,reply):
    music_reply = chatgpt('你是我的個人助理', get_music_name.format(result))
    reply_json = json.loads(music_reply)
    music_name = reply_json["music_name"]
    if music_name == 'None':
        music_reply  = '無法提取歌名'
    else:
        music_url = get_music_url(music_name)
        text_to_speech(f'正在為您播放{music_name}', "zh-tw")  # 文字轉語音檔 /uploads/temp.wav
        try:
            music_reply = f'正在為您播放{music_name}'
            youtube_to_wav(music_url)
        except Exception as e:
            reply = music_reply = '播放失敗'
            print(e)
            text_to_speech(reply, "zh-tw")  # 文字轉語音檔 /uploads/temp.wav
        save_chat(result, reply)  # 儲存對話
    return reply,music_reply

def translate(result):
    reply_j = chatgpt('你是一個專業的翻譯員', translate_language.format(result))
    reply_json = json.loads(reply_j)
    lang_param = reply_json["language"]
    try:  # 嘗試將翻譯後的字轉語音
        reply = chatgpt('你是一個專業的翻譯員', f'不要加入其他資訊，只輸出翻譯結果，請依照以下文句進行翻譯:{result}')
        text_to_speech(reply, lang_param)  # 文字轉語音檔 /uploads/temp.wav
    except:
        lang_param = 'zh-tw'
        reply = '翻譯錯誤'
    return reply, lang_param

def google_search(message1):
    message1.append({'role': 'user', 'content': template_google.format(result)})  # 餵給googlesearch樣板
    go_reply = chatgpt(None, None, message1)
    reply_json = json.loads(go_reply)
    search_Y_N = reply_json["search"]
    keyword = reply_json["keyword"]
    print('搜尋關鍵字:', keyword)
    if search_Y_N == 'Y':
        content = "最新資訊：\n"
        i_c = 1
        for item in search(keyword, advanced=True, num_results=5):
            content += f"標題：{item.title}\n"
            content += f"摘要：{item.description}\n\n"
        content += "只需單純用繁體中文回答以下問題，如果遇到科學單位請以台灣常用的為準(°C、kg、m/s等等)，不要加上額外資訊：\n"
        result_2 = content + result
    else:
        result_2 = result
    return result_2

translate_language = '''
請取出以下文句中翻譯語言的ISO 639-1的代號
```
{}
```
請以下列 JSON 格式回答我, 除了 JSON 格式資料外,
不要加上額外資訊：
```
{{
    "language":"ISO 639-1的代號"
}}
```
如果不會, 請以下列 JSON 格式回答我：
```
{{
    "language":"None"
}}
'''

function_choice = '''
需要開燈:t1,
白色燈:t1,
紅色燈:t2,
綠色燈:t3,
藍色燈:t4,
黃色燈:t5,
紫色燈:t6,
藍綠色燈:t7,
循環燈:t8,
彩色燈:t8,
需要關燈:t9,
需要播放音樂or想聽:t10,
需要翻譯(如:...的英文):t11,
高鐵或台鐵(火車):t12,
其他:None
請根據以下文句選擇符合的代號```{}```
並以下列 JSON 格式回答我, 除了 JSON 格式資料外,
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
請將以下文句中的歌曲名稱篩選出來,不要加上額外資訊```{}```
請以下列 JSON 格式回答我, 除了 JSON 格式資料外,不要加上額外資訊：
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


# =========================主程式=====================================================

load_dotenv()
env_key = "OPENAI_API_KEY"
if env_key in os.environ and os.environ[env_key]:
    openai.api_key = os.environ[env_key]

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

# 設定 led 代號
light_list = {'t1':'白色燈',
              't2':'紅色燈',
              't3':'綠色燈',
              't4':'藍色燈',
              't5':'黃色燈',
              't6':'紫色燈',
              't7':'藍綠色燈',
              't8':'循環燈',
              't9':'關燈'}
# 加入youtube下載套件
tool = YouTubeSearchTool()

app = Flask(__name__) # 建立 app

UPLOAD_FOLDER = path+'/uploads'  # 上傳資料的資料夾
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 根目錄
@app.route('/')
def hello():
    return "Welcome to the audio server!"

# 上傳PCM音檔
@app.route('/upload_audio', methods=['POST'])
def upload_audio():
    audio_data = request.data
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'input.pcm')
    with open(file_path, 'ab') as audio_file:
        audio_file.write(audio_data)
    return '上傳成功'

# 與 chatgpt 溝通
@app.route('/fix_audio', methods=['GET'])
def fix_audio():
    # =======================處理錄音檔並轉成文字=============================================
    result = speech_to_text()
    # result = input('user: ')
    result = '我想去名古屋日文'#我想聽枯萎的玫瑰
    # ======================判斷是否成功識別語音=========================================
    if result == '無法識別語音':
        print(f"Chatgpt：{result}")
        delete_file(f"{path}/uploads/temp.wav") # 刪除殘留語音檔 temp
        text_to_speech(result, "zh-tw")  # 文字轉語音檔 /uploads/temp.wav
        return f"${result}$"
    else:
        print(f"USER：{result}")
        # AI判別是否啟用功能或一般對話
        reply = chatgpt('你是一個智能助手',function_choice.format(result))
        # print(reply)
        reply_json = json.loads(reply)
        reply = reply_json["function_code"] # reply 會輸出 t1~t12 or None
        reply_text = reply
        lang_param = 'zh-tw'
        for i in range(12, 0,-1): # 從最後一筆開始是因為 t1 也包含在 t10、t11、t10,所以要從t12開始判斷
            if f't{i}' in reply: # 判斷是否要處理特定任務
                reply = f't{i}' # 讓 reply 設成代號
                # ==================================音樂撥放器========================================================
                if 't10' == reply:
                    reply, music_reply = player_machine(result,reply) # 取得要播放的歌名 music_reply，並下載音檔
                    reply_text = reply_information = reply_sound = music_reply
                # ========================================翻譯機======================================================
                elif 't11' == reply:
                    reply, lang_param = translate(result)
                    reply_text = reply_sound = reply
                    reply_information = ''
                # ========================================高鐵/台鐵時刻查詢=============================================
                elif 't12' == reply:
                    reply = Train_Module.find_best_train(result)
                    reply_information = ''
                    reply_text = reply_sound = reply
                # ========================================控制led燈====================================================
                else:
                    light = light_list[reply]

                    if reply == 't9': reply_light = f'已為您{light}' # 關燈
                    elif reply == 't8':reply_light = f'開啟{light}' # 循環燈 (因為是迴圈所以要先發聲再開燈)
                    else:reply_light = f'已為您開啟{light}' # 一般燈(先開燈再發聲)
                    reply_text = reply_information = reply_sound = reply_light

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
            result_2 = google_search(message)
            message.append({'role': 'user', 'content': result_2})  # 最後加上本次使用者訊息

            reply = chatgpt(messages_dict = message)  # chatgpt
            reply_text = reply_sound =reply
            reply_information = ''

        save_chat(result, reply)
        text_to_speech(reply_sound, lang_param)  # 文字轉語音檔 /uploads/temp.wav
        print(f"Chatgpt：{reply_text}")
        return f'{result}${reply}${reply_information}'
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
