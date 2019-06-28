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
#include "dtw.h"

#define ERROR (-1)
#define OK   0

class DnnForward
{
public:
	DnnForward():feat_handle(NULL),dnn_model(NULL),dnn_handle(NULL) { }
	
	~DnnForward()
	{
		freeAmScore();
	}

    int createAmScore(int feat_type,int frame_size,int frame_skip,int sample_rate,
            int channal_num,char *ammodel);
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

	int Reset()
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
	}
    /*
     * free acostics model correlation source.
     * */
    void freeAmScore();

private:
    struct feat_param_st feat_param;
    struct feat_st *feat_handle;
    struct Dnn_Feat dnn_feat; //some need free;

    struct dnn_para_t dnn_param;
    void *dnn_model;
    void *dnn_handle;
};

class DnnWakeupTemplate
{
private:
	// template space
	struct Template
	{
		float *_data;
		int _dims;
		int _frames;
		Template():_data(NULL),_dims(0),_frames(0){}
		~Template()
		{
			if(_data != NULL)  
				free(_data); // because _data is dnn out ,it's malloc,so free it.
			_data = NULL;
		}
	};
public:

    DnnWakeupTemplate():
		_forward(NULL),_template(NULL),_ntemp(0),_nth(0),
		_state(NULL),align_length(0),
		isstart(1),_cur_frame(0),_dtw(NULL),
		_distance_dtw(NULL),_ndist(0),
		_average(0),_deviation(0),
		dtw_thread(0),average_thread(0){ }
    ~DnnWakeupTemplate()
	{
		if(_forward != NULL)
			delete _forward;
		_forward = NULL;
		if(_template != NULL)
		{
			for(int i=0;i<_ntemp;++i)
			{
				if(_template[i]._data != NULL)
					free(_template[i]._data);
				_template[i]._data = NULL;
			}
			delete []_template;
			_template = NULL;
		}
		if(_compose_temp._data != NULL)
			free(_compose_temp._data);
		_compose_temp._data = NULL;

		if(_state != NULL)
			delete []_state;
		_state = NULL;
		delete _dtw;
		_dtw = NULL;
		delete []_distance_dtw; 
		_distance_dtw = NULL;
	}

	int Init(char *dnnmodel,char *templatename,int n);

	void StartWakeup();

	float ProcessData(char *data,int length);

	int ReadTemplate(char *filename);

	int WriteTemplate(char *filename);

	void ComposeTemplate(Template &A,Template &B,
			Align_res *align,int start,int end,float factor);

	int CreateTemplate(char *data,int len,int n);

	int AddTemplate(char *data,int len,int n);

private:
    /*
     * judge DTW alignment result.
     * return confidence coefficient,if more then 0,it's right,else wrong.
     * */
	float JudgeDtw();

	float ProcessWakeUp(char *data,int len,int flag);
private:

	char _templatename[256];
	DnnForward *_forward;
	Template *_template;
	int _ntemp;
	int _nth; // init 0

	Template _compose_temp;
	int *_state ;
    
	int align_length; //wave align length == wave affect length
    int isstart;      //whether it's the first frame,if yes isstart=1,else isstart=0;
    int _cur_frame;    //record current frame

	// dtw information
    DtwAlign *_dtw;
	float *_distance_dtw;
	int _ndist; // _ndist = _ntemp * (_ntemp-1)/2

	float _average;
	float _deviation;

    float dtw_thread;//0.0005
    float average_thread;
};

#endif
