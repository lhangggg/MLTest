
#include "opencv2\opencv.hpp"
#include <iostream>

using namespace std;
using namespace cv;
using namespace cv::ml;

int main(){
    cv::Mat img = imread("E:\\project\\MachineLearning\\KNN\\data\\digits.png");
    cv::Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    int band_width = 20;
    int m = gray.rows / band_width; //原图为1000*2000
    int n = gray.cols / band_width; //裁剪为5000个20*20的小图块

    cv::Mat data_set,labels_set; //特征矩阵
    for (int i = 0; i < n; i++){
        int offset_col = i*band_width; //列上的偏移量
        for (int j = 0; j < m; j++){
            int offset_row = j*band_width; //行上的偏移量
            //截取20*20的小块
            cv::Mat tmp;
            //变成一行：1*400
            gray(Range(offset_row, offset_row + band_width), 
                Range(offset_col, offset_col + band_width)).copyTo(tmp);
            data_set.push_back(tmp.reshape(0,1)); //序列化后放入特征矩阵
            //对应的标注，因为每五行一个数字
            labels_set.push_back((int)j / 5); 
        }
    }
    
    data_set.convertTo(data_set, CV_32F); //uchar型转换为cv_32f
    int samples_num = data_set.rows; //5000*400
    int train_num = 3000;
    cv::Mat train_data, train_labels;
    train_data = data_set(Range(0, train_num), Range::all()); //前3000个样本为训练数据
    train_labels = labels_set(Range(0, train_num), Range::all());


    //使用KNN算法, 数字的类别
    int K = 5;
    //降训练数据封装成一个TrainData对象，送入train函数
    Ptr<TrainData> tData = TrainData::create(train_data, ROW_SAMPLE, train_labels);
    Ptr<KNearest> model = KNearest::create();
    model->setDefaultK(K);
    model->setIsClassifier(true);
    model->train(tData);


    //预测分类
    double train_hr = 0, test_hr = 0;
    Mat response;
    // compute prediction error on train and test data
    for (int i = 0; i < samples_num; i++)
    {
        Mat sample = data_set.row(i);
        float r = model->predict(sample); //对所有行进行预测

        //预测结果与原结果相比，相等为1，不等为0
        r = std::abs(r - labels_set.at<int>(i)) <= FLT_EPSILON ? 1.f : 0.f;
        if (i < train_num)
            train_hr += r; //累积正确数
        else
            test_hr += r;        
    }
    test_hr /= samples_num - train_num;
    train_hr = train_num > 0 ? train_hr / train_num : 1.;
    printf("accuracy: train = %.1f%%, test = %.1f%%\n",
    train_hr*100., test_hr*100.);
    waitKey(0);
    return 0;
}