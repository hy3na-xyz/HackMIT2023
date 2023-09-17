from cortex2 import EmotivCortex2Client
import time
import tkinter as tk
import threading



timebetween = 50




count = 0
data = None
#constants
url = "wss://localhost:6868"
client = EmotivCortex2Client(url,
                                client_id='wguQFTerRM1Z59ERmmM7Tgiwybb743q9lHh6I6SF',
                                client_secret="E4I16rsg94u20uYpP2t5tgty9eOp9QbURSvOB7BcrP1qLBCwIsUM8cV8vN9jWz6Wxq7QOlViiH2ApYS772Fgdi6vKlYyDP0kLmM8WvopQtAzxXGOSbt9tjElKGeN1ety",
                                check_response=True,
                                authenticate=True,
                                debug=True)
def setup():
    global client
    global data
    global count
    client.authenticate()
    client.query_headsets()
    client.connect_headset(0)
    client.create_session(0)
    client.subscribe(streams=["eeg", "com"])
    a = client.subscriber_messages_handled
    time.sleep(5)
    b = client.subscriber_messages_handled
    print((b - a) / 5)
    while True:
        print(list(client.data_streams.values()))
        time.sleep(0.1)
        data = (list(client.data_streams.values()))
        count = count + 1
    



firsttime = True
def start():
    global firsttime
    global data
    global root
    global count
    root = tk.Tk()
    root.title("EEG data")


    
    canvas = tk.Canvas(root, width=400, height=200)
    canvas.pack()
    eeg_label = tk.Label(root, text="EEG Data: ")
    counter = tk.Label(root, text="Counter: ")


    window_width = root.winfo_screenwidth()
    window_height = root.winfo_screenheight()

    def switch_color_left():
        canvas.create_rectangle(0, 0, window_width, window_height, fill="green")
        canvas.create_rectangle(0, 0, 200, window_height, fill="red")
    def switch_color_right():
        canvas.create_rectangle(0, 0, window_width, window_height, fill="red")
        canvas.create_rectangle(0, 0, 200, window_height, fill="green")    

    

    
    eeg_label.pack()
    counter.pack()
    def getdata():
        global data
        global count
        #print("This is the data being sent to client")
        #print(data)
        root.after(100, getdata)
        if data != None:
            try:
                data = data[0]['eeg'][0]['data']
            except:
                pass
        data = str(data)
        eeg_label.config(text=f"EEG Data: {data}")
        counter.config(text=f"Counter: {count}")
        if (count<50):
            switch_color_left()
        else:
            switch_color_right()
            if count> 100:
                count = 0
        return data

    getdata()    
    root.mainloop()

def main():
    eegThread = threading.Thread(target=setup)
    tkinterThread = threading.Thread(target=start)

    eegThread.start()
    tkinterThread.start()
    
main()