import glob

dataset_dir = "/home/gaofeng/datasets/formatted_data/kitti/"
# test_files = "/home/gaofeng/workspace/ai_2018fall/final_project/struct2depth/test_files_eigen.txt"

def get_test_files():
    f = open(test_files, 'r')
    lines = f.readlines()
    test_files_list = []
    for line in lines:
        seqname = line.split('/')[1]
        lr = line.split('/')[2].split('_')[-1]
        new_seqname = seqname + '_' + lr
        frame_id = line.split('/')[-1].split('.')[0]
        test_files_list.append(new_seqname + ' ' + frame_id)
    return test_files_list

def main():
    # test_files_list = get_test_files()
    f = open(dataset_dir + 'train_dvo.txt', 'w')
    for d in glob.glob(dataset_dir + '/*/'):
        seqname = d.split('/')[-2]
        d2 = glob.glob(d + '/*.jpg')
        d2 = sorted(d2)
        for line_ in d2:
            frameid = line_.split('/')[-1].split('.')[0]
            line = seqname + ' ' + frameid
            f.write(line + '\n')
    f.close()

if __name__ == '__main__':
    main()
