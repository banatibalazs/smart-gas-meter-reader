import pandas as pd
from flask import Flask, request, render_template, url_for, send_from_directory, redirect
from markupsafe import escape
import os
import pandas
import glob
from flask import g
import datetime
# import json
from flask import json

app = Flask(__name__, static_folder='static')

QUOTA = {'01': 19, #JAN
         '02': 16, #FEB
         '03': 14, #MAR
         '04': 8, #APR
         '05': 3, #MAY
         '06': 1, #JUN
         '07': 1, #JUL
         '08': 1, #AUG
         '09': 2, #SEP
         '10': 6, #OCT
         '11': 12, #NOV
         '12': 17 #DEC
         }

# def save_graph(df):
#     df = df.sort_values(by='datetimes', ascending=True)
#     plt.rcParams['font.size'] = '12'
#     plt.figure(figsize=(15, 10))
#     plt.scatter(df["datetimes"], df["values"])
#     plt.grid()
#     plt.xlabel('Dátum', fontsize=26)
#     plt.ylabel('Értékek', fontsize=26)
#     plt.savefig('static/graphs/graph.png')
def get_monthly_stats(data_df):
    a = data_df.sort_values(by='datetimes').groupby(['year', 'month'])['values'].agg(['first', 'last'])
    a = a.diff(axis=1).to_dict()
    a = a['last']
    monthly_keys = [list(key) for key in a.keys()]
    monthly_values = list(a.values())
    return monthly_keys, monthly_values


def load_data():
    images = glob.glob('static/images/*.jpg')
    fnames = [image.split('/')[-1] for image in images]
    values = [int(fname.split('_')[1]) for fname in fnames]
    datetimes = [datetime.datetime.fromtimestamp(int(fname.split('_')[-1].split('.')[0])) for fname in fnames]
    data_df = pd.DataFrame(list(zip(images, fnames, datetimes, values)),
                columns =['images', 'fnames', 'datetimes', 'values'])
    data_df['datetimes_str'] = data_df['datetimes'].dt.strftime('%Y-%m-%d %H:%M:%S')
    data_df['year'] = data_df['datetimes'].dt.strftime('%Y')
    data_df['month'] = data_df['datetimes'].dt.strftime('%m')
    data_df = data_df.sort_values(by='datetimes', ascending=False)
    return data_df

def get_dates(df):
    st_year = df['datetimes'].iloc[-1].year
    st_month = df['datetimes'].iloc[-1].month
    st_day = df['datetimes'].iloc[-1].day

    if st_month < 10:
        st_month = '0' + str(st_month)

    if st_day < 10:
        st_day = '0' + str(st_day)

    en_year = df['datetimes'].iloc[0].year
    en_month = df['datetimes'].iloc[0].month
    en_day = df['datetimes'].iloc[0].day

    if en_month < 10:
        en_month = '0' + str(en_month)

    if en_day < 10:
        en_day = '0' + str(en_day)
    start_date = f"{st_year}-{st_month}-{st_day}"
    end_date = f"{en_year}-{en_month}-{en_day}"
    return start_date, end_date

def filter_data_by_dates(start_date, end_date, data_df):
    _start_date = datetime.datetime.fromisoformat(start_date)
    _end_date = datetime.datetime.fromisoformat(end_date + 'T' + '23:59:59')
    filtered_df = data_df[(data_df['datetimes'] >= _start_date) & (data_df['datetimes'] <= _end_date)]
    return filtered_df


@app.route("/", methods=['GET', 'POST'])
def index():
    data_df = load_data()
    start_date, end_date = get_dates(data_df)

    if request.method == 'POST':
        return "<p>Hello, World!</p><br><p>POST</p>"
    elif request.method == 'GET':

        monthly_keys, monthly_values = get_monthly_stats(data_df)
        return render_template('index.html',
                               first_image=data_df['images'].tolist()[-1],
                               first_v=data_df['values'].tolist()[-1],
                               first_d=data_df['datetimes'].tolist()[-1],
                               last_image=data_df['images'].tolist()[0],
                               last_v=data_df['values'].tolist()[0],
                               last_d=data_df['datetimes'].tolist()[0],
                               start_date=start_date,
                               end_date=end_date,
                               monthly_keys=monthly_keys,
                               monthly_values=monthly_values,
                               QUOTA=QUOTA
                               )
    else:
        return "<p>Neither POST, nor GET</p>"

@app.route('/filter', methods=['GET', 'POST'])
def example():
    if request.method == 'POST':
        start_date = request.form.get('startDate')
        end_date = request.form.get('endDate')
    else:
        start_date = request.args.get('startDate')
        end_date = request.args.get('endDate')

    data_df = load_data()
    filtered_df = filter_data_by_dates(start_date, end_date, data_df)

    if filtered_df.shape[0] <= 0 or filtered_df.shape[0] is None:
        return redirect('/')

    values = filtered_df['values'].tolist()
    images = filtered_df['images'].tolist()
    datetimes = filtered_df['datetimes'].tolist()

    monthly_keys, monthly_values = get_monthly_stats(data_df)
    return render_template('index.html',
                           first_image=images[-1],
                           first_v=values[-1],
                           first_d=datetimes[-1],
                           last_image=images[0],
                           last_v=values[0],
                           last_d=datetimes[0],
                           start_date=start_date,
                           end_date=end_date,
                           monthly_keys=monthly_keys,
                           monthly_values=monthly_values,
                           QUOTA=QUOTA
                           )

@app.route('/dateRequest', methods=['GET', 'POST'])
def dateRequest():
    start_date = request.args.get('startDate')
    end_date = request.args.get('endDate')

    data_df = load_data()
    filtered_df = filter_data_by_dates(start_date, end_date, data_df)
    # filtered_df = filtered_df.sort_values(by='datetimes')

    if filtered_df.shape[0] <= 0 or filtered_df.shape[0] is None:
        return redirect('/')

    values = filtered_df['values'].tolist()
    images = filtered_df['images'].tolist()
    datetimes = filtered_df['datetimes'].tolist()

    first_date = datetimes[-1].date()
    first_time = datetimes[-1].time()
    last_date = datetimes[0].date()
    last_time = datetimes[0].time()

    first_value = values[-1]
    last_value = values[0]

    time_delta = str((last_date - first_date).days + 1)
    value_delta = last_value - first_value
    average_value = round(value_delta / int(time_delta), 2)

    # print(first_date)
    # print(first_time)
    # print(last_date)
    # print(last_time)

    data = {
        'first_image': images[-1],
        'first_v': first_value,
        'first_date': str(first_date),
        'first_time': str(first_time),
        'last_image': images[0],
        'last_v=': last_value,
        'last_date': str(last_date),
        'last_time': str(last_time),
        'time_delta': time_delta,
        'value_delta': value_delta,
        'average_value': average_value
        }

    response = app.response_class(
        response=json.dumps(data),
        status=200,
        mimetype='application/json'
    )
    return response
    # return [images[-1], values[-1], datetimes[-1], images[0], values[0], datetimes[0]]



@app.route('/changeRequest', methods=['GET', 'POST'])
def changeRequest():
    new_value = request.args.get('value')
    old_img_name = request.args.get('img')
    # 1122_13710_424_166782693.jpg  <-- index_5value_3value_timestamp.jpg
    old_img_name_slices = old_img_name.split('_')

    print('index:', old_img_name_slices[0])
    print('value:', old_img_name_slices[1])
    print('value:', old_img_name_slices[2])
    print(new_value)

    new_img_name = old_img_name_slices[0] + '_' + str(new_value) + '_' + old_img_name_slices[2] + '_' + old_img_name_slices[3]
    print('old:', old_img_name)
    print('new:', new_img_name)
    folder = "./static/images/"
    os.rename(folder + old_img_name, folder + new_img_name)

    response = app.response_class(
        response=json.dumps([old_img_name, new_img_name]),
        status=200,
        mimetype='application/json'
    )
    return response

#
# @app.route('/page', methods=['GET', 'POST'])
# def page():
#     if request.method == 'GET':
#         page = int(request.args.get('page'))
#         start_date = request.args.get('startDate')
#         end_date = request.args.get('endDate')
#         elPerPage = int(request.args.get('elPerPage'))
#
#         from_idx = (page-1)*elPerPage
#         to_idx = page*elPerPage
#
#         data_df = load_data()
#         filtered_df = filter_data_by_dates(start_date, end_date, data_df)
#         page_num = filtered_df.shape[0] // elPerPage
#         if filtered_df.shape[0] % elPerPage != 0:
#             page_num += 1
#
#         temp_df = filtered_df.iloc[from_idx:to_idx]
#
#         values = temp_df['values'].tolist()
#         images = temp_df['images'].tolist()
#         fnames = temp_df['fnames'].tolist()
#         datetimes = temp_df['datetimes'].tolist()
#         datetimes_str = temp_df['datetimes_str'].tolist()
#
#         interval_values = filtered_df['values'].tolist()
#         interval_dates = filtered_df['datetimes'].tolist()
#
#         monthly_keys, monthly_values = get_monthly_stats(data_df)
#         return render_template('index_old.html',
#                                images=images,
#                                fnames=fnames,
#                                dates=datetimes,
#                                values=values,
#                                list_v=json.dumps(values),
#                                list_d=json.dumps(datetimes_str),
#                                first_image=filtered_df['images'].tolist()[-1],
#                                first_v=interval_values[-1],
#                                first_d=interval_dates[-1],
#                                last_image=filtered_df['images'].tolist()[0],
#                                last_v=interval_values[0],
#                                last_d=interval_dates[0],
#                                start_date=start_date,
#                                end_date=end_date,
#                                elPerPage=elPerPage,
#                                page_num=page_num,
#                                page=page,
#                                monthly_keys=monthly_keys,
#                                monthly_values=monthly_values,
#                                QUOTA=QUOTA
#                                )
#     else:
#         return redirect('/')


if __name__ == "__main__":
    app.run(host="0.0.0.0")