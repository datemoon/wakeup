
#1include <stdio.h>

int main()
{
  printf("%s:---res===",argv[1]);
	int sil_state = 0;
	int sta[] = {0,1,0,2,0,3,0,4,0};
	float wei[] = {0.08,1,0.1,1,0.08,1,0.1,1,0.08};
	vector<int> states(sta,sta+9);
	vector<float> weights(wei, wei+9);
	int max_frame_len = 200;
	WakeupSearch wakeup(sil_state ,false);
	wakeup.Init(states,weights,max_frame_len);
//	wakeup.ProcessData(probs,frontend->plp_flen-5,outdim);
	
	int add_frames = 15;
	for (int i=0;i<frontend->plp_flen-5 ;i+=add_frames )
	{
		wakeup.ProcessData(probs+outdim*i, add_frames,outdim);
	}

	printf("\n");
	return 0;
}
