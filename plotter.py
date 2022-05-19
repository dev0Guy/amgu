
import pickle
import matplotlib.pyplot as plt
import numpy as np

file_name = "WhiteUntargeted"
with open(f"res/{file_name}_results.pkl","rb") as f:
    results = pickle.load(f)
    for idx, res in enumerate(results):
        fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(40,100))
        
        imgs = res['obs']
        axs[0,0].imshow(imgs[0].T)
        axs[0,0].set_title('Original State - 0')
        axs[0,1].imshow(imgs[1].T)
        axs[0,1].set_title('Original State - 1')
        print(imgs[2].shape)
        axs[1,0].imshow(imgs[2].T)
        axs[1,0].set_title('Original State - 2')
        axs[1,1].imshow(imgs[3].T)
        axs[1,1].set_title('Original State - 3')
        
        obs_pre = res['obs_preprocess']
        label = res['action']
        adv_label = res['adv_action']

        print(obs_pre.shape)
        axs[0,2].imshow(obs_pre.T)
        axs[0,2].set_title(f'Obs after preprocess : {label}')
        adv_imgs = res['adv_obs']
        axs[1,2].imshow(adv_imgs.T)
        axs[1,2].set_title(f'Adversrial Attack State: {adv_label}')


        # labels = res['action']
        # axs[1,0].imshow(labels)
        # # axs[1,0].imshow(np.expand_dims(labels[idx].cpu().detach().numpy(),axis=0))
        # axs[1,0].set_title('The output orinial model')
        # axs[1,0].set(xlabel='Choosen action:'+str(int(labels)))

        # adv_labels = res['adv_action']
        # axs[1,1].imshow(adv_labels)
        # axs[1,1].set_title('The output adversarial model')
        # axs[1,1].set(xlabel='Choosen action:'+str(adv_labels))

        # # print(torch.argmax(labels[idx]))
        # # print(torch.argmax(adv_labels[idx]))
        fig.savefig(f'res/{file_name}/Prototype_{idx}.png')



# # imports
# import plotly.graph_objs as go
# import plotly.express as px
# import pandas as pd
# import numpy as np
# import json

# PATHS = ['evaluation/CNN/DQN_Single-CityFlow_d0374_00000_0_2022-05-16_12-36-31_checkpoint-200.json']
# KEYS = ['rewards', 'ATT', 'QL']

# def pre_plot(list_paths, key):
#   plot_name = f'{key} for each episodes'
#   dict_data = {}
#   for path in list_paths:
#     file_name = (path.split('/')[-1]).split('_')
#     line_name = file_name[0]+" "+file_name[6]+" "+file_name[2]
#     with open(path, 'rb') as file:
#       pkl = json.load(file)
#       dict_data[line_name] = pkl[key] # save the specific data according to key
#   return plot_name, dict_data


# def plot(paths, keys):
#     plot_name, dict_data = pre_plot(paths, keys[1])

#     # sample data in a pandas dataframe
#     # np.random.seed(1)
#     df=pd.DataFrame(dict_data)
#     # df = df.cumsum()

#     # define colors as a list 
#     colors = px.colors.qualitative.Antique

#     # convert plotly hex colors to rgba to enable transparency adjustments
#     def hex_rgba(hex, transparency):
#         col_hex = hex.lstrip('#')
#         col_rgb = list(int(col_hex[i:i+2], 16) for i in (0, 2, 4))
#         col_rgb.extend([transparency])
#         areacol = tuple(col_rgb)
#         return areacol

#     rgba = [hex_rgba(c, transparency=0.2) for c in colors]
#     colCycle = ['rgba'+str(elem) for elem in rgba]

#     # Make sure the colors run in cycles if there are more lines than colors
#     def next_col(cols):
#         while True:
#             for col in cols:
#                 yield col
#     line_color=next_col(cols=colCycle)

#     # plotly  figure
#     fig = go.Figure()

#     # add line and shaded area for each series and standards deviation
#     for i, col in enumerate(df):
#         new_col = next(line_color)
#         x = list(df.index.values+1)
#         y1 = df[col]
#         y1_upper = [(y + np.std(df[col])) for y in df[col]]
#         y1_lower = [(y - np.std(df[col])) for y in df[col]]
#         y1_lower = y1_lower[::-1]

#         # standard deviation area
#         fig.add_traces(go.Scatter(x=x+x[::-1],
#                                     y=y1_upper+y1_lower,
#                                     fill='tozerox',
#                                     fillcolor=new_col,
#                                     line=dict(color='rgba(255,255,255,0)'),
#                                     showlegend=False,
#                                     name=col))

#         # line trace
#         fig.add_traces(go.Scatter(x=x,
#                                 y=y1,
#                                 line=dict(color=new_col, width=2.5),
#                                 mode='lines',
#                                 name=col))
#     # set x-axis
#     fig.update_layout(autosize=False,
#         width=1500,
#         height=500, 
#         title= plot_name,
#         xaxis=dict(range=[1,len(df)]))

#     fig.savefig('evaluation/CNN')

# plot(PATHS, KEYS)