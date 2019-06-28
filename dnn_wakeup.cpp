/*
 * Copyright 2015-2016
 *
 * File Type : C++ Source File(*.cpp)
 * File Name : worddnn_wakeup.cpp
 * Module    :
 * Create on :2015/12/21
 * Author    :hubo
 *
 * This file realization class DnnWakeupModel .
 * */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <math.h>
#include "dnn_wakeup.h"

//#define RES_LOG

static float VectorDistance(float *A,float *B,int n)
{
	float tot=0;
	for(int i=0;i<n;++i)
		tot += (A[i]-B[i]) * (A[i]-B[i]);
	return sqrtf(tot/n);
}

static void DtwAddVecVec(float *A,float*B,int n)
{
	for(int i=0;i<n;++i)
		A[i] += B[i];
}

static void DtwScaleVec(float *A,int n,float factor)
{
	for(int i=0;i<n;++i)
		A[i] /= factor;
}


void DnnForward::freeAmScore()
{
    if(dnn_feat.feats != NULL){    
        free(dnn_feat.feats);
        dnn_feat.feats=NULL;
    }
    destory_feature(feat_handle);
    destroy_dnn_scoring_handle(dnn_handle);
    destroy_dnn_scoring_res(dnn_model);
}

int DnnForward::createAmScore(int feat_type,int frame_size,int frame_skip,int sample_rate,int channal_num,char *ammodel)
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
int DnnForward::processDataDnn(char *data,int len,float **out,int *dim,int *col,int flag)
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

int DnnWakeupTemplate::Init(char *dnnmodel,char *templatename,int n)
{
	_forward = new DnnForward;
	_forward->createAmScore(FB,25,10,8000,40,dnnmodel);
	strcpy(_templatename, templatename);
	_template = new Template[n];
	_ntemp = n;
	_ndist = _ntemp * (_ntemp-1) / 2;
	_distance_dtw = new float[_ndist];
	return OK;
}

int DnnWakeupTemplate::AddTemplate(char *data,int len,int n)
{
	float *output=NULL;
	int dnn_out_dim=0;
	int dnn_out_col=0;
	int rtu = _forward->processDataDnn(data,len, &output, &dnn_out_dim,&dnn_out_col,3);
    if(rtu == -1)
        return ERROR;
	_template[n]._data = output;
	_template[n]._dims = dnn_out_dim;
	_template[n]._frames = dnn_out_col;
	_forward->Reset();
	return OK;
}

// n start from 0
int DnnWakeupTemplate::CreateTemplate(char *data,int len,int n)
{
	if(n >= _ntemp)
		return ERROR;
	int ret = AddTemplate(data,len,n);

	if(n > 0)
	{
		int end = n *(n+1)/2-1;
		// start register, calculate dtw distance
		for(int i=n-1; i>=0;--i)
		{
			int col = _template[i]._frames;
			int row = _template[n]._frames;

			int dim = _template[n]._dims;
			_dtw = new DtwAlign(row, col);
			int *state = new int[col];
			for(int m=0;m<col;++m)
				state[m] = m;
			_dtw->Init(state,col);
			for(int r=0;r<row;++r)
			{
				for(int c=0;c<col;++c)
				{
					float *A = _template[i]._data+c*dim;
					float *B = _template[n]._data+r*dim;

					float dist = VectorDistance(A,B,dim);
					_dtw->setData2Dtw(dist);
				}
			}
			int tot_align = row+col;
			int align_length=0;
			Align_res* align = _dtw->StartAlign(&align_length);
			int first_align = tot_align-align_length;
			_distance_dtw[end--] = (align[tot_align-1].total_score - align[first_align].total_score)/align_length;

			fprintf(stderr,"%f \n",_distance_dtw[end+1]);
			if(i == 0)// template compound
			{ // all template compose to first template
				if(_compose_temp._data == NULL)
				{
					_compose_temp._dims = _template[0]._dims;
					_compose_temp._frames = _template[0]._frames;
					_compose_temp._data = (float*)malloc(sizeof(float) * _compose_temp._dims * _compose_temp._frames);
					if(NULL == _compose_temp._data)
					{
						fprintf(stderr,"malloc failed!\n");
						return -1;
					}
					memcpy(_compose_temp._data,_template[0]._data,sizeof(float) * _compose_temp._dims * _compose_temp._frames);
				}
				ComposeTemplate(_compose_temp,_template[n],align,first_align,tot_align, 1.0);
			}
			delete _dtw;
			_dtw = NULL;
			delete []state;
		}
	}

	// judge register whether success
	if(n == _ntemp-1)
	{
		for(int i=0;i<_ndist;++i)
			_average += _distance_dtw[i];
		_average /= _ndist;
		for(int i=0;i<_ndist;++i)
			_deviation += powf(_distance_dtw[i] - _average,2.0);
		_deviation = sqrtf(_deviation/_ndist);
		if(_average > 0.1)
		{
			fprintf(stderr,"%s %d template train failed.\n",__FILE__,__LINE__);
			return -2;// register failed.
		}
		fprintf(stderr,"average %f,deviation%f\n",_average,_deviation);
	}
	return OK;
}

void DnnWakeupTemplate::ComposeTemplate(Template &A,Template &B,
		Align_res *align,int start,int end,float factor)
{
	for(int i=start;i<end;)
	{
		int cur_state = align[i].state;
		int next_state = -1;
		if(i < end-1)
			next_state = align[i+1].state;
		int cur_frame = align[i].frame;
		int dim = A._dims;
		int n = 1;
		float *Adata = NULL;
		float *Bdata = NULL;
		while(next_state == cur_state || n == 1)
		{
			n++;
			Adata = A._data + cur_state * dim;
			Bdata = B._data + cur_frame * dim;
			DtwAddVecVec(Adata,Bdata,dim);
			i++;
			if(i == end)
				break;
			cur_frame = align[i].frame;
			if(i < end-1)
				next_state = align[i+1].state;
		}
		DtwScaleVec(Adata,dim,(float)n);
	}
}

int DnnWakeupTemplate::WriteTemplate(char *filename)
{
	FILE *fp = fopen(filename,"w");
	if(NULL == fp)
	{
		fprintf(stderr,"fopen %s error.\n",filename);
		return ERROR;
	}
	if(1 != fwrite(&_average,sizeof(float),1,fp))
	{
		fprintf(stderr,"write average %s error.\n",filename);
		return ERROR;
	}
	if(1 != fwrite(&_deviation,sizeof(float),1,fp))
	{
		fprintf(stderr,"write deviation %s error.\n",filename);
		return ERROR;
	}
	if(1 != fwrite(&_compose_temp._dims,sizeof(int),1,fp))
	{
		fprintf(stderr,"write dims %s error.\n",filename);
		return ERROR;
	}
	if(1 != fwrite(&_compose_temp._frames,sizeof(int),1,fp))
	{
		fprintf(stderr,"write frames %s error.\n",filename);
		return ERROR;
	}
	if(_compose_temp._dims*_compose_temp._frames != fwrite(_compose_temp._data,sizeof(float),
				_compose_temp._dims*_compose_temp._frames,fp))
	{
		fprintf(stderr,"write data %s error.\n",filename);
		return ERROR;
	}
	fclose(fp);
	return OK;
}

int DnnWakeupTemplate::ReadTemplate(char *filename)
{
	FILE *fp = fopen(filename,"rb");
	if(NULL == fp)
	{
		fprintf(stderr,"fopen %s error.\n",filename);
		return ERROR;
	}
	if(1 != fread(&_average,sizeof(float),1,fp))
	{
		fprintf(stderr,"read average %s error.\n",filename);
		return ERROR;
	}
	if(1 != fread(&_deviation,sizeof(float),1,fp))
	{
		fprintf(stderr,"read deviation %s error.\n",filename);
		return ERROR;
	}
	if(1 != fread(&_compose_temp._dims,sizeof(int),1,fp))
	{
		return ERROR;
	}
	if(1 != fread(&_compose_temp._frames,sizeof(int),1,fp))
	{
		return ERROR;
	}
	if(_compose_temp._dims*_compose_temp._frames != fread(_compose_temp._data,sizeof(float),
				_compose_temp._dims*_compose_temp._frames,fp))
	{
		return ERROR;
	}
	fclose(fp);
	return OK;
}

//flag=0,start else 2
//return 0:continue ; -1 error; 1 sucess.
float DnnWakeupTemplate::ProcessWakeUp(char *data,int len,int flag)
{
	if(_dtw == NULL)
	{
		int col = _compose_temp._frames;
		int row = col *3/2;
		_dtw = new DtwAlign(row, col);
		_state = new int[col];
		for(int i=0;i<col;++i)
			_state[i] = i;
		_dtw->Init(_state,col);
	}
	float *output=NULL;
	int dnn_out_dim=0;
	int dnn_out_frames=0;
	int rtu = _forward->processDataDnn(data,len, &output, &dnn_out_dim,&dnn_out_frames,flag);
	if(rtu == 1)
		return 0;
	else if(rtu == -1)
		return -1;
	if(dnn_out_frames > 0)
	{
		int col = _compose_temp._frames;
		int dim = _compose_temp._dims;
		for(int r = 0;r<dnn_out_frames;++r)
		{
			for(int c = 0;c<col;++c)
			{
				float *A = _compose_temp._data+c*dim;
				float *B = output + r*dim;
				float dist = VectorDistance(A,B,dim);
				_dtw->setData2Dtw(dist);
			}
		}
	}
	_cur_frame += dnn_out_frames;
	free(output);

	if(_cur_frame > 30)
	{
		float average_dist = 0;
		if((int)(_cur_frame/10) > (int)(_cur_frame-dnn_out_frames)/10 
				&& (average_dist = JudgeDtw())< (_average - _deviation))
		{
			printf("frame %d wav_length %d score %f OK\n",_cur_frame,50,average_dist);
			_cur_frame=0;
			return average_dist;
		}
	}
	return 0;
}
	
//
float DnnWakeupTemplate::JudgeDtw()
{
    int dtw_len=0;
    Align_res *dtw_ali = _dtw->StartAlign(&dtw_len);
    if(dtw_ali == NULL)
        return 0;
	int end = _dtw->GetAlignLength();

	float tot_score = dtw_ali[end].total_score/dtw_len;
    return tot_score;
}

void DnnWakeupTemplate::StartWakeup()
{
	_forward->Reset();
    isstart=1;
    _cur_frame=0;
    if(_dtw != NULL)
		_dtw->Reset();
}

float DnnWakeupTemplate::ProcessData(char *data,int length)
{
    int flag = 0;
    if(isstart == 1){
        isstart=0;
        flag=0;
    }
    else if(isstart == 0){
        flag=2;
    }
    return ProcessWakeUp(data,length,flag);
}
