/*
 * Copyright 2015-2016
 *
 * File Type : C++ Header File(*.h)
 * File Name : worddnn_wakeup.h
 * Module    :
 * Create on :2015/12/21
 * Author    :hubo
 *
 * This class is used wake up the strategy.
 * */


#ifndef __WORDDNN_WAKUP_H__
#define __WORDDNN_WAKUP_H__

#include "dnn_score_api.h"
#include "feat_api.h"
#include "base.h"
#include "dtw.h"

#define ERROR (-1)
#define OK   0

class WordDnnWakeupModel:public CModel
{
public:

    WordDnnWakeupModel();
    ~WordDnnWakeupModel();

    virtual int createModel(char *model,int max_frame_len);

    virtual int startWakeup();

    virtual float processData(char *data,int length);

    virtual float processFeature(float *data,int length);
private:
    int createAmScore(int feat_type,int frame_size,int frame_skip,int sample_rate,
            int channal_num,char *ammodel);

    /* 
     * @data    :int put wave data
     * @len     :data length
     * @flag    :0 is start else 2
     * return 0 : continue ; 
     *        -1: error;
     *        >0: sucess.
     */
    float processWakeUp(char *data,int len,int flag);

    /*
     * @data:input wav data
     * @len :data length
     * @out :dnn output, interial malloc and external free
     * @dim :out dim
     * @col :out column
     * @flag:0 start, 1 end ,2 middle,3 start+end
     * return: 0 : success ;
     *         -1: fail ; 
     *         1 : data isn't enough
     */
    int processDataDnn(char *data,int len,float **out,int *dim,int *col,int flag);

    /*
     * free acostics model correlation source.
     * */
    void freeAmScore();

    /*
     * judge DTW alignment result.
     * return confidence coefficient,if more then 0,it's right,else wrong.
     * */
    float judgeDtw();

private:
    struct feat_param_st feat_param;
    struct feat_st *feat_handle;
    struct Dnn_Feat dnn_feat; //some need free;

    struct dnn_para_t dnn_param;
    void *dnn_model;
    void *dnn_handle;
//judge relate parameter
    int align_length; //wave align length == wave affect length
    int isstart;      //whether it's the first frame,if yes isstart=1,else isstart=0;
    int cur_frame;    //record current frame
//    int last_gt_times;//use for the last state score greater then second last state score ; if (last state score greater)<(second last state score);last_gt_times++;if last_gt_times>5 ;then continue judge
//    int last_gt_flag; //use for last_gt_times recount.

    //DP(dtw) need space and source

    DtwAlign *dtw;
    Align_State *align_word;
    int align_word_num;
    float prev_score;
    float dtw_thread;//0.0005
    float average_thread;
};

#endif
