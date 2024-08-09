import glob
import datetime
from tflite_support.task import core
from tflite_support.task import processor
from mqtt_server_and_analyzer.helper_functions import *
import cv2
from tqdm import tqdm


dim = math.ceil(math.sqrt(len(skipped_images)))
    rows = dim
    cols = dim
    fig = plt.figure(figsize=[15, 18])

    for i, img_path in enumerate(skipped_images):
        img = cv2.imread(img_path)
        ax = plt.subplot(rows, cols, (i + 1))
        plt.xticks([])
        plt.yticks([])
        plt.imshow(img / 255.)
    plt.show()

    df = pd.read_csv('./result_2/result_images.csv', header=None)
    df_sorted = df.sort_values(by=0)
    MIN_VAL = 13641876
    MAX_VAL = 13800000
    df_sorted = df_sorted[df_sorted[1] > MIN_VAL]
    df_sorted = df_sorted[df_sorted[1] < MAX_VAL]
    df_sorted[1] = df_sorted[1].apply(lambda x: x / 1000.)
    plt.plot(list(df_sorted[1]))
    plt.show()


    def timestamp_from_date(string) -> int:
        daymonth, time = string[:-10].split('T')
        year, month, day = daymonth.split('-')
        h, m, sms = time.split(':')
        s, ms = sms.split('_')
        x_datetime = datetime.datetime(int(year), int(month), int(day), int(h), int(m), int(s), int(ms))
        return datetime.datetime.timestamp(x_datetime)


    tu = list(zip(df_sorted[0], list(df_sorted[1])))

    clean = []
    for i, t in enumerate(tu):
        if i == 0 or i == len(tu) - 1:
            continue
        else:
            current = tu[i][1]
            left_nb = tu[i - 1][1]
            right_nb = tu[i + 1][1]

            diff_left = current - left_nb
            diff_right = right_nb - current
            if diff_left >= 0 and diff_left < 1 and \
                    diff_right >= 0 and diff_right < 1:
                clean.append(t)

    filenames, results = zip(*clean)
    print(len(results), "---", len(filenames), "---", results[0], "---", filenames[0])

    df_cleaned = pd.DataFrame()
    df_cleaned["filenames"] = filenames
    df_cleaned["measured_values"] = results

    df_cleaned["stamp"] = df_cleaned["filenames"].apply(timestamp_from_date)
    df_cleaned["year"] = df_cleaned["stamp"].apply(lambda x: datetime.datetime.fromtimestamp(x).year)
    df_cleaned["month"] = df_cleaned["stamp"].apply(lambda x: datetime.datetime.fromtimestamp(x).month)
    df_cleaned["day"] = df_cleaned["stamp"].apply(lambda x: datetime.datetime.fromtimestamp(x).day)

    df_cleaned["hour"] = df_cleaned["stamp"].apply(lambda x: datetime.datetime.fromtimestamp(x).hour)
    df_cleaned["minute"] = df_cleaned["stamp"].apply(lambda x: datetime.datetime.fromtimestamp(x).minute)
    df_cleaned["second"] = df_cleaned["stamp"].apply(lambda x: datetime.datetime.fromtimestamp(x).second)

    df_cleaned["ms"] = df_cleaned["stamp"].apply(lambda x: datetime.datetime.fromtimestamp(x).microsecond)

    stamps = list(df_cleaned['stamp'])
    start_date = datetime.datetime.fromtimestamp(stamps[0])
    end_date = datetime.datetime.fromtimestamp(stamps[-1])

    date_diff = end_date - start_date
    daylines = [datetime.datetime.timestamp(start_date) + i * 86400 for i in range(date_diff.days + 3)]

    df_cleaned['datetime'] = df_cleaned['stamp'].apply(lambda x: datetime.datetime.fromtimestamp(x))
    df_cleaned['date'] = df_cleaned['stamp'].apply(lambda x: datetime.date.fromtimestamp(x))

    plt.figure(figsize=(15, 10))
    plt.scatter(df_cleaned['date'], df_cleaned['measured_values'])
    # for line in daylines:
    #     plt.axvline(x = line, color = 'gray', label = 'axvline - full height')
    #     plt.text(10.1,0,'blah',rotation=90)
    plt.grid()
    plt.xlabel('Date')
    plt.ylabel('values')
    plt.show()

    plt.figure(figsize=(15, 10))
    plt.scatter(df_cleaned['datetime'], df_cleaned['measured_values'])
    for line in daylines:
        plt.axvline(x = line, color = 'gray', label = 'axvline - full height')
        plt.text(10.1,0,'blah',rotation=90)
    plt.grid()
    plt.xlabel('Date')
    plt.ylabel('values')
    plt.show()