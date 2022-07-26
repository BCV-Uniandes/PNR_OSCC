
#files = 'ego4d_swin_16cls_multigpu_E20.json'
#json_files=glob('*.json')
def evaluate_keyframe_distance(file):
    import json
    import numpy as np 
    print('evaluating file: '+file)
    f= open(file)  
    inference =  json.load(f) 
    preds=[res[0] for i,res in enumerate(inference) if i%2==0 ]
    #samp_rate=[res[0] for i,res in enumerate(inference) if i%2!=0 ]
    f.close()

    with open('data/ego4d/ego4d_val_rawframes.txt','r+') as f:
        lines=f.read().splitlines()
    strip_list = [line.replace('\n','').split(' ') for line in lines if line != '\n']
    video_infos = [{'name':ann[0],'offset':int(ann[1]),'duration':int(ann[2]),'label':int(ann[3])} for ann in strip_list]
    gt_labels = [ann['label']- ann['offset'] for ann in video_infos]
    samp_rate = [(a['duration']/16) for a in video_infos]

    _preds_indv = np.argmax(preds,axis=1)
    frame_preds=np.round(np.array(samp_rate)*(_preds_indv))
    err_frame = abs(gt_labels-frame_preds)
    fps=30
    err_secs = np.mean(err_frame/fps)
    print(f'error: {err_secs}')

#for f in json_files: 
#evaluate_keyframe_distance(files) 

#normalized_label = [(ann['label']- ann['offset'])/ann['duration'] for ann in video_infos]
#err_seconds = err_frame/fps 
#plt.scatter(normalized_label,err_seconds,linewidths=0.5)

#df = pd.DataFrame(list(zip(err_seconds,err_seconds_baseline)),columns = ['VidSwinT','Baseline'],index=names)
#df.sort_values(by=['VidSwinT','Baseline'],ascending=[True,False])
#one: 10d22bd7-d9b8-46ea-8302-5f9b0ea0884d-348_1210286-356_1210286 gt: 57 vid: 57 baseline: 68 offset: 10517
#two: 68a6332f-9d17-4d86-8855-b85867e72412-52_166666666666664-60_166666666666664 gt: 98 vid: 98 baseline: 101 offset: 1580
#three: 5a64bf10-8d3b-46fd-b9d7-a96e12ec1bce-1879_0210286-1887_0210286 gt: 98 vid: 98 baseline: 101 offset: 56385

#idx = np.where([name == '7be77d68-0113-4782-b9a7-2afecde59418-31_4-39_4'for name in names])[0][0]
#gt, frame_preds, offset
#Best and worst for VidSwinT
#VidSwinT    0.00
#Baseline    0.05
#66 66.0 29577 150
#Name: 67cfb244-4c31-4090-b14c-bf015fea5db0-983_9333333333333-991_9333333333333, dtype: float64


#idx = np.where([name == 'a4e9f10d-b2ce-4e7d-b4cb-3b1017d42265-764_3876952666667-772_3876952666667'for name in names])[0][0]
#print(gt_labels[idx],frame_preds[idx],video_infos[idx]['offset'],video_infos[idx]['duration'])

# VidSwinT    0.000000
# Baseline    0.051667
#70 70.0 15023 159
# Name: bb19a930-5671-4165-b012-6ef888c7fd39-498_8-506_8, dtype: float64

# VidSwinT    4.266667
# Baseline    4.156667
#230 102.0 9632 234
# Name: ed6ede34-2134-427f-ae4c-ff5b554b6596-320_9-328_9, dtype: float64

# VidSwinT    4.066667
# Baseline    3.970000
#219 97.0 22949 222
# Name: a4e9f10d-b2ce-4e7d-b4cb-3b1017d42265-764_3876952666667-772_3876952666667, dtype: float64

def test_inference(file):
    import json
    import numpy as np 
    print('evaluating file: '+file)
    f= open(file)  
    inference =  json.load(f) 
    preds=[res[0] for i,res in enumerate(inference) if i%2==0 ]
    #samp_rate=[res[0] for i,res in enumerate(inference) if i%2!=0 ]
    f.close()

    with open('data/ego4d/ego4d_test_rawframes_zero.txt','r+') as f:
        lines=f.read().splitlines()
    strip_list = [line.replace('\n','').split(' ') for line in lines if line != '\n']
    video_infos = [{'name':ann[0],'offset':int(ann[1]),'duration':int(ann[2])} for ann in strip_list]
    samp_rate = [(a['duration']/16) for a in video_infos]

    _preds_indv = np.argmax(preds,axis=1)
    frame_preds=np.round(np.array(samp_rate)*(_preds_indv)) 

    inference = [{"unique_id":video_infos[i]['name'],"pnr_frame":frame_preds[i]} for i,vid in enumerate(video_infos)]

    with open('Inference_'+file, 'w') as outfile:
        json.dump(inference, outfile)
  

# sampling_rates = [(a['duration']/16) for a in video_infos] 
# frame_preds=np.round(np.array(sampling_rates)*(_preds_indv)) 

# middle = [8*samp for samp in sampling_rates] 

# err_frame = abs(gt_labels-np.array(middle)) 
# fps=30 
# err_secs = np.mean(err_frame/fps)
# print(err_secs)  

# plt.figure()
# n, bins, patches = plt.hist(np.array(gt_labels),bins=60) 
# plt.title('Relative PNR')
# plt.xlabel('Relative frame number')
# plt.ylabel('Frequency (# of frames)')
# plt.show() 
# plt.savefig('histogram_val_60bins.png')
# plt.close()

# abs_pnr = [ann['label'] for ann in video_infos]
# plt.figure()
# n, bins, patches = plt.hist(np.array(abs_pnr),bins=60) 
# plt.title('Absolute PNR')
# plt.xlabel('Absolute frame number')
# plt.ylabel('Frequency (# of frames)')
# plt.show() 
# plt.savefig('absolute_histogram_val_60bins.png')
# plt.close()

# normalized_label = [(ann['label']- ann['offset'])/ann['duration'] for ann in video_infos]
# plt.figure()
# n, bins, patches = plt.hist(np.array(normalized_label),bins=60) 
# plt.title('Normalized PNR')
# plt.xlabel('percentage of video')
# plt.ylabel('Frequency (# of frames)')
# plt.show() 
# plt.savefig('normalized_histogram_val_60bins.png')
# plt.close()

# middle_label = [(ann['duration'])*middle for ann in video_infos]
# err_frame = abs(gt_labels-np.array(middle_label)) 
# fps=30 
# err_secs = np.mean(err_frame/fps)
# print(err_secs)

# normalized_label = [(ann['label']- ann['offset'])/ann['duration'] for ann in video_infos]
# plt.figure()
# n, bins, patches = plt.hist(np.array(normalized_label),bins=10) 
# plt.title('Normalized PNR')
# plt.xlabel('percentage of video')
# plt.ylabel('Frequency (# of frames)')
# plt.show() 
# plt.savefig('normalized_histogram_val_10bins.png')
# plt.close()


#     with open('data/ego4d/ego4d_train_rawframes.txt','r+') as f:
#         lines=f.read().splitlines()
#     strip_list = [line.replace('\n','').split(' ') for line in lines if line != '\n']
#     video_infos = [{'name':ann[0],'offset':int(ann[1]),'duration':int(ann[2]),'label':int(ann[3])} for ann in strip_list]
#     gt_labels = [ann['label']- ann['offset'] for ann in video_infos]

# plt.figure()
# n, bins, patches = plt.hist(np.array(gt_labels),bins=60) 
# plt.title('Relative PNR')
# plt.xlabel('Relative frame number')
# plt.ylabel('Frequency (# of frames)')
# plt.show() 
# plt.savefig('histogram_train_60bins.png')
# plt.close()

# abs_pnr = [ann['label'] for ann in video_infos]
# plt.figure()
# n, bins, patches = plt.hist(np.array(abs_pnr),bins=60) 
# plt.title('Absolute PNR')
# plt.xlabel('Absolute frame number')
# plt.ylabel('Frequency (# of frames)')
# plt.show() 
# plt.savefig('absolute_histogram_train_60bins.png')
# plt.close()

# normalized_label = [(ann['label']- ann['offset'])/ann['duration'] for ann in video_infos]
# plt.figure()
# n, bins, patches = plt.hist(np.array(normalized_label),bins=60) 
# plt.title('Normalized PNR')
# plt.xlabel('percentage of video')
# plt.ylabel('Frequency (# of frames)')
# plt.show() 
# plt.savefig('normalized_histogram_train_60bins.png')
# plt.close()

# normalized_label = [(ann['label']- ann['offset'])/ann['duration'] for ann in video_infos]
# plt.figure()
# n, bins, patches = plt.hist(np.array(normalized_label),bins=10) 
# plt.title('Normalized PNR')
# plt.xlabel('percentage of video')
# plt.ylabel('Frequency (# of frames)')
# plt.show() 
# plt.savefig('normalized_histogram_train_10bins.png')
# plt.close()

# #EValuation code that gave us 0.6713307677001673 temporal error
# #frame_preds = [ann['duration']*0.45 for ann in video_infos] 
# #inference = [{"unique_id":video_infos[i]['name'],"pnr_frame":frame_preds[i]} for i,vid in enumerate(video_infos)] 

# ## Calculate how many of each class for training. 
# import json
# import numpy as np 

# with open('data/ego4d/ego4d_train_rawframes.txt','r+') as f:
#     lines=f.read().splitlines()
# strip_list = [line.replace('\n','').split(' ') for line in lines if line != '\n']
# video_infos = [{'name':ann[0],'offset':int(ann[1]),'duration':int(ann[2]),'label':int(ann[3])} for ann in strip_list]
# gt_labels = [ann['label']- ann['offset'] for ann in video_infos]
# categories=np.round(np.array(gt_labels)/15).astype(int)
# categories_freq = collections.OrderedDict(sorted(Counter(categories).items()))
# freq_values =  np.array(list(categories_freq.values())) 
# weights = 1/freq_values

# [4.44444444e-03, 3.26797386e-03, 3.48432056e-03, 4.73933649e-03, 3.21543408e-03, 2.50000000e-03, 1.49253731e-03, 3.56760614e-04, 1.05719421e-04, 2.81848929e-04, 1.05152471e-03, 2.63852243e-03, 4.54545455e-03, 8.13008130e-03, 1.21951220e-02, 1.72413793e-02, 1.25000000e-01]

#     with open('test_inference_sr.json', 'w') as outfile:
#         json.dump(inference, outfile)