/*
 * Copyright 2015-2016
 *
 * File Type : C++ Source File(*.cpp)
 * File Name : worddnn_wakeup.cpp
 * Module    :
 * Create on :2015/12/21
 * Author    :hubo
 *
 * This file realization class WordDnnWakeupModel .
 * */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "worddnn_wakeup.h"

//#define RES_LOG

WordDnnWakeupModel::WordDnnWakeupModel()
{
    feat_handle=NULL;
    dnn_model=NULL;
    dnn_handle=NULL;

    align_length = 0;
    isstart=1;
    cur_frame=0;
    prev_score = 0;

    dtw=NULL;
    align_word=NULL;
    align_word_num=0;
}

void WordDnnWakeupModel::freeAmScore()
{
    if(dnn_feat.feats != NULL){    
        free(dnn_feat.feats);
        dnn_feat.feats=NULL;
    }
    destory_feature(feat_handle);
    destroy_dnn_scoring_handle(dnn_handle);
    destroy_dnn_scoring_res(dnn_model);
}

WordDnnWakeupModel::~WordDnnWakeupModel()
{
    delete dtw;
    delete[] align_word;
    freeAmScore();
}

#define STATE_NUM 9
int WordDnnWakeupModel::createModel(char *model,int max_frame_len)
{
    int queue_state[STATE_NUM]={0,1,2,0,3,4,5,6,0};
    dtw = new DtwAlign(max_frame_len,STATE_NUM);
    if(0 != dtw->Init(queue_state,STATE_NUM)){
        fprintf(stderr,"Init error!\n");
        return -1;
    }
    align_word_num = dtw->getStateDim();
    align_word = new Align_State[align_word_num];
    return createAmScore(FB,25,10,8000,40,model);
};

int WordDnnWakeupModel::createAmScore(int feat_type,int frame_size,int frame_skip,int sample_rate,int channal_num,char *ammodel)
{
    feat_param.type = feat_type;
    feat_param.frame_size = frame_size;
    feat_param.frame_skip = frame_skip;
    feat_param.sample_rate = sample_rate;

    feat_handle = create_feature(&(feat_param));
    if(NULL == feat_handle){
        fprintf(stderr,"create_feature failed!\n");
        return -1;
    }

    dnn_feat.feats = NULL;
    dnn_feat.dim = 0;
    dnn_feat.col = 0;
    dnn_feat.start = 1;
    dnn_feat.end = 0;
    dnn_feat.capacity = 0;

    memset(dnn_param.szNnet,0x00,sizeof(dnn_param.szNnet));
    memcpy(dnn_param.szNnet,ammodel,strlen(ammodel));
    dnn_param.nFrameDim = channal_num;

    dnn_model = create_dnn_scoring_res(&(dnn_param));
    if(dnn_model == NULL){
        fprintf(stderr,"create_dnn_scoring_res error!\n");
        return -1;
    }

    dnn_handle = create_dnn_scoring_handle(dnn_model);
    if(dnn_handle == NULL){
        fprintf(stderr,"create_dnn_scoring_handle error!\n");
        return -1;
    }
    return 0;
}

//data:input wav data
//len : data length
//out :dnn output, interial malloc and external free 
//dim :out dim
//col :out column
//flag:0 start, 1 end ,2 middle,3 start+end
//return: 0: success ;-1: fail ; 1: data isn't enough
int WordDnnWakeupModel::processDataDnn(char *data,int len,float **out,int *dim,int *col,int flag)
{
    *dim = 0;
    *col = 0;
    int feat_dim=0;
    int feat_col=0;
    api_set_data(feat_handle,data,len);
    float *feats = api_get_feature(feat_handle,&feat_col,flag%2);
    if(feats == NULL && flag != 1 && flag != 3)
        return 1;

    feat_dim = get_feature_dim(feat_handle);

    *dim = dnn_param.nStates;
    if(flag == 0){    
        dnn_feat.start = 1;
        dnn_feat.end =0;
    }
    else if(flag == 1){
        dnn_feat.start = 0;
        dnn_feat.end = 1;
    }
    else if(flag == 3){
        dnn_feat.start = 1;
        dnn_feat.end = 1;
    }
    if(0 > dnn_score(dnn_handle,&dnn_feat,feats ,
                feat_col,feat_dim,out,*dim,col,0)){
        fprintf(stderr,"dnn_score error!\n");
        return -1;
    }
    if(feats != NULL)
        free(feats);
    feats = NULL;
    return 0;
}

//
float WordDnnWakeupModel::judgeDtw()
{
    int dtw_len=0;
    Align_res *dtw_ali = dtw->startAlign(&dtw_len);
    if(dtw_ali == NULL)
        return 0;
    int st_num=0;
    int *st = dtw->getStates(&st_num);
    if(st_num != align_word_num){
        fprintf(stderr,"source error!\n");
        return -1;
    }
    judge_align_res(align_word,dtw_ali,dtw_len,st,st_num);
    {//next judge align right or wrong.
        int start_frame =0,end_frame =0;
        int total_nonsil_frame=0;
        float total_score = 0;
        int i=0;
        int word_len = 0;
        for(i=0;i<align_word_num;++i){
            if(align_word[i].state != 0){
                word_len = align_word[i].end-align_word[i].start+1;
                if(start_frame == 0)
                    start_frame = align_word[i].start;
                end_frame = align_word[i].end;
                total_nonsil_frame += (align_word[i].end-align_word[i].start+1);
                total_score += align_word[i].state_score;
                if( (align_word[i].end-align_word[i].start+1) < 5 
                        && align_word[i].state != 3)//&& align_word[i].state != 4)
                {
                    return 0;
                }
            }
        }
        //if(prev_score > total_score/total_nonsil_frame)
        {
            if(total_score/total_nonsil_frame > 0.1
                    && total_nonsil_frame > 60 )//&& word_len > 10 )
            {
                align_length = end_frame-start_frame;
                return total_score/total_nonsil_frame;
            }
        }
        prev_score = total_score/total_nonsil_frame;
    }
    return 0;
}

//flag=0,start else 2
//return 0:continue ; -1 error; 1 sucess.
float WordDnnWakeupModel::processWakeUp(char *data,int len,int flag)
{
    float *output=NULL;
    int dnn_out_dim=0;
    int dnn_out_col=0;
    int rtu = processDataDnn(data,len, &output, &dnn_out_dim,&dnn_out_col,flag);
    if(rtu == 1)
        return 0;
    else if(rtu == -1)
        return -1;
    if(dnn_out_col >0){
        int state_dim = 0;
        int *states = dtw->getStates(&state_dim);

        int i=0,j=0;
        for(i=0;i<dnn_out_col;++i){
#ifdef RES_LOG
            printf("%d: ",cur_frame+i);
#endif
            for(j=0;j<state_dim;++j){
                float factor=1;
                if(states[j] == 0)
                    factor=0.008;
                dtw->setData2Dtw(output[dnn_out_dim*i+states[j]]*factor);
#ifdef RES_LOG
                printf("%f ", output[dnn_out_dim*i+states[j]]);
#endif
            }
#ifdef RES_LOG
            printf("\n");fflush(stdout);
#endif
        }//for
        cur_frame += dnn_out_col;
        free(output);
        if(cur_frame > 50){
            float average_posterior = 0;
            if((int)(cur_frame/10) > (int)(cur_frame-dnn_out_col)/10 
                    && (average_posterior = judgeDtw())>0){
#ifdef RES_LOG
                printf("frame %d wav_length %d OK\n",cur_frame,align_length);
#endif
                cur_frame=0;
//                last_gt_times = 0;
                return average_posterior;
            }
        }
    }
    return 0;
}


int WordDnnWakeupModel::startWakeup()
{
    int final_col;
    api_get_feature(feat_handle,&final_col,1);
    if(dnn_feat.feats != NULL)
        free(dnn_feat.feats);
    dnn_feat.feats = NULL;
    dnn_feat.dim = 0;
    dnn_feat.col = 0;
    dnn_feat.start = 1;
    dnn_feat.end = 0;
    dnn_feat.capacity = 0;
    prev_score = 0;
    isstart=1;
    cur_frame=0;
    dtw->Reset();
    return align_length;
}

float WordDnnWakeupModel::processData(char *data,int length)
{
    int flag = 0;
    if(isstart == 1){
        isstart=0;
        flag=0;
    }
    else if(isstart == 0){
        flag=2;
    }
    return processWakeUp(data,length,flag);
}

float WordDnnWakeupModel::processFeature(float *data,int length)
{
    return 0;
}

