#include <stdio.h>
#include <sys/stat.h>
#include <stdlib.h>
#include "dnn_wakeup.h"

#define FILENUM 3

int main(int argc,char *argv[])
{
	if(argc != 6 && argc != 7)
	{
		fprintf(stderr,"%s dnnmodel template file1 file2 file3 testfile\n",argv[0]);
		return -1;
	}
	char *dnnmodel = argv[1];
	char *templatename = argv[2];

	DnnWakeupTemplate wakeup;
	int ret = wakeup.Init(dnnmodel,templatename,FILENUM);
	if(ret != 0)
	{
		fprintf(stderr,"init failed.\n");
		return 0;
	}
	for(int i=0;i<FILENUM;++i)
	{
		char *audio_name = argv[i+3];
		struct stat ast;
		ret = stat(audio_name, &ast);
		if (ret != 0)
		{
			fprintf(stderr, "stat file fail: %s", audio_name);
			return -1;
		}
		int audio_size = ast.st_size;
		FILE* audio_file = fopen(audio_name, "rb");
		if (audio_file == NULL)
		{
			fprintf(stderr, "open file fail: %s", audio_name);
			return -1;
		}
		char *buf = (char *)malloc(sizeof(char)*audio_size);

		ret = fread(buf, sizeof(char), audio_size, audio_file);
		if (ret != audio_size)
		{
			fprintf(stderr, "read file %s fail: readsize: %d, need size: %d",
					audio_name, (int) ret, (int) ast.st_size);
			return -1;
		}
		fclose(audio_file);

		ret = wakeup.CreateTemplate(buf ,audio_size ,i);
		if(ret != 0)
		{
			fprintf(stderr,"CreateTemplate failed.\n");
			return -1;
		}
		free(buf);
	}


	// verify
	if(argc == 7)
	{
		printf("verifg:\n");
		char *audio_name = argv[6];
		struct stat ast;
		ret = stat(audio_name, &ast);
		if (ret != 0)
		{
			fprintf(stderr, "stat file fail: %s", audio_name);
			return -1;
		}
		int audio_size = ast.st_size;
		FILE* audio_file = fopen(audio_name, "rb");
		if (audio_file == NULL)
		{
			fprintf(stderr, "open file fail: %s", audio_name);
			return -1;
		}
		char *buf = (char *)malloc(sizeof(char)*audio_size);

		ret = fread(buf, sizeof(char), audio_size, audio_file);
		if (ret != audio_size)
		{
			fprintf(stderr, "read file %s fail: readsize: %d, need size: %d",
					audio_name, (int) ret, (int) ast.st_size);
			return -1;
		}
		fclose(audio_file);
		float score = 0;
#ifdef STREAM
		int len = 400;
		for(int i=0;i+len <audio_size;i += len)
		{
			score = wakeup.ProcessData(buf+i ,len);
			if(score > 0)
				break;
		}
#else
		score = wakeup.ProcessData(buf ,audio_size);
#endif
		printf("score is %f\n",score);
		free(buf);
	}	

	ret = wakeup.WriteTemplate(templatename);
	if(ret != 0)
	{
		fprintf(stderr,"WriteTemplate failed.\n");
		return -1;
	}
	return 0;
}
