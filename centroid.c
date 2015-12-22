#include<stdio.h>
#include<string.h>
#include<stdlib.h>
#include<math.h>
#include<sys/time.h>
#include<assert.h>
#define max(a,b) a>b?a:b
#define min(a,b) a>b?b:a
#define SWAP(x,y) do { typeof(x) SWAP = x;x=y;y=SWAP;} while(0)

struct SparseData
{
	int *document;
	int *featureid;
	float *freq;
	int nonZeros;
	int maxFeatureId;
};

struct Data
{
	struct SparseData csr;

	int isCSR;	
	int *Y;
	int *mask;
	int features;
	char **feature_label;
	char **class_label;

	int observations;
	int isSparse;
	int totalClasses;
};

struct CENTROIDParameters
{
	int K;
	float **centroids;
	float **negcentroids;
};

struct rlabel
{
	int docid;
	int fileid;
	char class_name[32];
};

struct Predictions
{
	int id;
	int label;
	float score;
};

struct Results
{
	float *f1_score;
	float *best_score;
	int *best_label;
	int test_observations;	
};

void merge(struct Predictions *distances,struct Predictions *l,int lc,struct Predictions *r,int rc)
{
	int i=0;
	int j=0;
	int k=0;

	i=0;
	j=0;
	k=0;

	while(i<lc && j < rc)
	{
		if(l[i].score < r[j].score)
		{
			distances[k].score = l[i].score;
			distances[k].id = l[i].id;
			distances[k].label = l[i].label;
			i++;
		}
		else
		{
			distances[k].score = r[j].score;
			distances[k].id = r[j].id;
			distances[k].label = r[j].label;
			j++;
		}
		k++;
	}
	while(i<lc)
	{
		distances[k].score = l[i].score;
		distances[k].id = l[i].id;
		distances[k].label = l[i].label;
		i++;
		k++;
	}
		
	while(j<rc)
	{
		distances[k].score = r[j].score;
		distances[k].id = r[j].id;
		distances[k].label = r[j].label;
		j++;
		k++;
	}
}

void mergeSort(struct Predictions *distances,int n)
{
	if(n<2) return;	
	int m = n/2;
	int i=0;
	
	struct Predictions *l;
	struct Predictions *r;

	l = (struct Predictions *)calloc(m,sizeof(struct Predictions));
	r = (struct Predictions *)calloc(n-m,sizeof(struct Predictions));

	for(i=0;i<m;i++) 
	{
		l[i].score = distances[i].score;
		l[i].id = distances[i].id;
		l[i].label = distances[i].label;
	}	
	
	for(i=m;i<n;i++)
	{
		r[i-m].score = distances[i].score;
		r[i-m].id = distances[i].id;
		r[i-m].label = distances[i].label;
	}

	mergeSort(l,m);
	mergeSort(r,n-m);
	merge(distances,l,m,r,n-m);
	free(l);
	free(r);
}

void sort(struct Predictions *distances,int n,int order)
{
	mergeSort(distances,n);
	if(1==order)
	{
		int i=0;
		for(i=0;i<n/2;i++)
		{
			SWAP(distances[i].score,distances[n-i-1].score);
			SWAP(distances[i].id,distances[n-i-1].id);
			SWAP(distances[i].label,distances[n-i-1].label);
		}
	}
}

void printInputData(struct Data *data)
{
	int i=0;
	int j=0;
    for(i=0;i<data->observations;i++)
    {
        for(j=data->csr.document[i];j<data->csr.document[i+1];j++)
            printf("(%d %.6f) ",data->csr.featureid[j],data->csr.freq[j]);
        puts("");
    }
}

struct Data readInput(char *fname)
{
	const int NSIZE=32;
	int i=0;
	int j=0;
	int dim=0;
	int featureid=0;
	int documentid=0;
	int observations=0;	
	int nonzeros=0;
	int prev_document=-1;
	int rowptr=0;
	int colptr=0;
	float freq=0.0;

	struct Data data;
	FILE* fp = fopen(fname,"r");
	if(!fp) return data;
	while(fscanf(fp,"%d%d%f",&documentid,&featureid,&freq)!=EOF)
	{
		nonzeros=nonzeros+1;
		observations = max(observations,documentid);
		dim = max(dim,featureid);
	}	
	fclose(fp);
	
	data.observations = observations;
	data.features = dim;
	data.csr.maxFeatureId = dim;
	data.csr.nonZeros = nonzeros;

	data.csr.document = (int *)calloc(observations+1,sizeof(int));
	data.csr.featureid = (int *)calloc(nonzeros+1,sizeof(int));
	data.csr.freq = (float *)calloc(nonzeros+1,sizeof(float));
	
	data.mask = (int *)calloc(observations+1,sizeof(int));
	data.Y = (int *)calloc(observations+1,sizeof(int));
	data.feature_label = (char**)calloc(dim+1,sizeof(char*));
	data.class_label = (char**)calloc(data.observations+1,sizeof(char*));
	
	data.isCSR=1;
	data.isSparse=1;	
	for(i=0;i<dim;i++) data.feature_label[i] = (char *)calloc(NSIZE,sizeof(char*));	
	for(i=0;i<data.observations;i++) data.class_label[i] = (char*)calloc(NSIZE,sizeof(char*));

	fp = fopen(fname,"r");
	if(!fp) return data;
	
	while(fscanf(fp,"%d%d%f",&documentid,&featureid,&freq)!=EOF)
	{
		if(prev_document!=documentid) 
		{
			data.csr.document[rowptr] = colptr;
			rowptr=rowptr+1;
		}
		prev_document = documentid;
		data.csr.featureid[colptr] = featureid-1;
		data.csr.freq[colptr] = freq;
		colptr = colptr+1;
	}
	data.csr.document[rowptr] = colptr;
	fclose(fp);
	return data;
}

struct rlabel* readrlabel(char *fname,int n)
{
	FILE *fp = fopen(fname,"r");
	if(!fp) return NULL;

	int i=0;
	struct rlabel *data = (struct rlabel*)calloc(n+1,sizeof(struct rlabel));
	while(fscanf(fp,"%d%s%d",&data[i].docid,data[i].class_name,&data[i].fileid)!=EOF) i++;
	fclose(fp);
	return data;
}

void readTrainFile(struct Data *data,char *fname)
{
	int docid=0;
	FILE *fp = fopen(fname,"r");
	if(!fp) return;
	while(fscanf(fp,"%d",&docid)!=EOF) data->mask[docid-1] =1;
	fclose(fp);
}

void readTestFile(struct Data *data,char *fname)
{
	int docid=0;
	FILE *fp = fopen(fname,"r");
	if(!fp) return;
	while(fscanf(fp,"%d",&docid)!=EOF) data->mask[docid-1] =0;
	fclose(fp);
}

void readClassFile(struct Data *data,char *fname)
{
	FILE *fp = fopen(fname,"r");
	if(!fp) return;

	int docid=0;
	int classid=-1;
	const int NSIZE=64;
	char *prev_class = (char *)calloc(NSIZE,sizeof(char));	
	char *class_name = (char *)calloc(NSIZE,sizeof(char));

	while(fscanf(fp,"%d%s",&docid,class_name)!=EOF)
	{
		if(strcmp(class_name,prev_class)) 
		{
				classid=classid+1;
				strcpy(data->class_label[classid],class_name);
		}
		data->Y[docid-1] = classid;
		strcpy(prev_class,class_name);
	}
	data->totalClasses = classid+1;
	fclose(fp);
	free(class_name);
	free(prev_class);
}

void readFeaturelabel(struct Data *data,char *fname)
{
	int i=0;
	FILE *fp = fopen(fname,"r");
	if(!fp) return;
	while(fscanf(fp,"%s",data->feature_label[i])!=EOF) i++;
	fclose(fp);
}

void normalizeVector(float *inp,int n)
{
	int i=0;
	int j=0;
	float csum=0;

	for(i=0;i<n;i++) csum = csum + inp[i]*inp[i];
	csum = sqrt(csum);
	for(i=0;i<n;i++) inp[i]/=csum;
}

void convert(struct Data *data,int format)
{
	int i=0;
	int j=0;
	int required_docs=0;
	int *feature = (int*)calloc(data->features,sizeof(int));
	for(i=0;i<data->features;i++) feature[i] = 1; //Laplace estimate

	for(i=0;i<data->observations;i++)
	{
		if(data->mask[i]==1) //considering only training document
		{
			required_docs = required_docs+1;
			for(j=data->csr.document[i];j<data->csr.document[i+1];j++)
				feature[data->csr.featureid[j]] = feature[data->csr.featureid[j]] + 1;
		}
	}
	for(i=0;i<data->observations;i++)
	{
		for(j=data->csr.document[i];j<data->csr.document[i+1];j++)
		{
			if(1 == format) data->csr.freq[j] = 1.0;
			else if(2 == format) data->csr.freq[j] = sqrt(data->csr.freq[i]);
			else if(3 == format) data->csr.freq[j] = log((float)required_docs/feature[data->csr.featureid[j]]);
			else if(4 == format) data->csr.freq[j] = sqrt(data->csr.freq[i]) * log((float)required_docs/feature[data->csr.featureid[j]]);
			else if(5 == format) data->csr.freq[j] = data->csr.freq[j] * log((float)required_docs/feature[data->csr.featureid[j]]);
		}
	}
	free(feature);
}

void featureTransformation(struct Data *data,char *type)
{
	if(!strcmp(type,"binary")) convert(data,1);
	if(!strcmp(type,"sqrt")) convert(data,2);
	if(!strcmp(type,"binaryidf")) convert(data,3); 
	if(!strcmp(type,"sqrtidf")) convert(data,4);
	if(!strcmp(type,"tfidf")) convert(data,5);
}

float getSparseSimilarity(float *centroid,int *feature,float *values, int st,int et)
{
	float csum=0.0;
	int i=st;
	while(i<et)
	{
		csum = csum + centroid[feature[i]]*values[i];
		i++;
	}
	return csum;
}

void Normalize(struct Data *data,int mask)
{
	int i=0;
	int j=0;
	float mu=0.0;
	float std=0.0;
    float csum = 0.0;

	for(i=0;i<data->observations;i++)
	{
        if(mask==data->mask[i])
        {
    	    csum = 0.0;
    	    for(j=data->csr.document[i];j<data->csr.document[i+1];j++) csum = csum + data->csr.freq[j]*data->csr.freq[j];
    	    csum = sqrt(csum);
    	    for(j=data->csr.document[i];j<data->csr.document[i+1];j++) data->csr.freq[j] /= csum;
	    }
    }
}

void initResults(struct Results *results,int totalClasses,int observations,int test_observations)
{
	results->f1_score = (float *)calloc(totalClasses,sizeof(float));
	results->best_label = (int *)calloc(observations+1,sizeof(int));
	results->best_score = (float *)calloc(observations+1,sizeof(float));
	results->test_observations = test_observations;
}

void setF1Score(struct Results *results,struct Predictions *pred,int observations,int iter)
{
	int i=0,j=0,label=0,tp=0,fp=0,fn=0;
	float f1 = 0.0,prec=0.0,recall=0.0,best_f1=0.0,f=0.0;

	for(i=0;i<observations;i++) if(pred[i].label == iter) fn = fn + 1;
	for(i=0;i<observations;i++)
	{
		if(pred[i].label == iter) {tp = tp+1;fn = fn-1;}
		else fp = fp + 1;

		prec = (fp+tp == 0 ? 0 : (float)tp/(fp+tp));
		recall = (fn+tp == 0 ? 0 : (float)tp/(fn+tp));
		f1 = (2*prec*recall)/(prec+recall);
		if(f1>best_f1) best_f1 = f1;

		if(results->best_score[pred[i].id] < pred[i].score)
		{
			results->best_score[pred[i].id] = pred[i].score;
			results->best_label[pred[i].id] = iter;
		}
	}
	results->f1_score[iter] = best_f1;
}

struct Results makePredictionsCENTROID(struct Data *data,struct CENTROIDParameters *params)
{
	int i=0,j=0,k=0,label=0,ptr=0,tp=0,fp=0,fn=0,test_observations=0;
	float sim=0.0,f1=0.0,prec=0.0,recall=0.0,best_f1=0.0;

	for(i=0;i<data->observations;i++) 
		if(data->mask[i]==0) 
			test_observations = test_observations + 1;

	struct Predictions *pred = (struct Predictions *)calloc(test_observations+1,sizeof(struct Predictions));
	struct Results results;
	initResults(&results,data->totalClasses,data->observations,test_observations);

	results.test_observations = test_observations;
	
	for(k=0;k<data->observations;k++) results.best_score[k] = -1.000;

	for(k=0;k<data->totalClasses;k++)
	{
		ptr = 0;
		for(i=0;i<data->observations;i++)
		{
			if(data->mask[i]==0)
			{
				sim = getSparseSimilarity(params->centroids[k],data->csr.featureid,data->csr.freq,data->csr.document[i],data->csr.document[i+1]);
				sim = sim - getSparseSimilarity(params->negcentroids[k],data->csr.featureid,data->csr.freq,data->csr.document[i],data->csr.document[i+1]);
				
				pred[ptr].id = i;
				pred[ptr].score = sim;
				pred[ptr].label = data->Y[i];
				ptr = ptr + 1;
			}
		}

		sort(pred,ptr,1);
		setF1Score(&results,pred,ptr,k);
	}

	if(pred) free(pred);
	return results;
}

struct CENTROIDParameters CENTROIDClassifier(struct Data *data)
{
	int i=0;
	int j=0;
	int p=0;

	struct CENTROIDParameters params;
	params.K = data->totalClasses;

	params.centroids = (float **)calloc(data->totalClasses,sizeof(float*));
	params.negcentroids = (float **)calloc(data->totalClasses,sizeof(float*));

	for(i=0;i<data->totalClasses;i++) 
	{
		params.centroids[i] = (float *)calloc(data->features,sizeof(float));
		params.negcentroids[i] = (float *)calloc(data->features,sizeof(float));
	}

	int *freq;
	freq = (int*)calloc(data->observations,sizeof(int));
	int observations = 0;

	for(i=0;i<data->observations;i++)
	{
		if(data->mask[i]==1)
		{
			observations =  observations + 1;
			freq[data->Y[i]] = freq[data->Y[i]]+1;

			for(j=data->csr.document[i];j<data->csr.document[i+1];j++)
		    		params.centroids[data->Y[i]][data->csr.featureid[j]] = params.centroids[data->Y[i]][data->csr.featureid[j]] + data->csr.freq[j];
        	
        	for(p=0;p<data->totalClasses;p++)
        	{
        		if(p==freq[data->Y[i]]) continue;
        		for(j=data->csr.document[i];j<data->csr.document[i+1];j++)
		    		params.negcentroids[p][data->csr.featureid[j]] = params.negcentroids[p][data->csr.featureid[j]] + data->csr.freq[j];
        	}
        }
    }

    for(i=0;i<params.K;i++)
    {
		for(j=0;j<data->features;j++) params.centroids[i][j] /= freq[data->Y[i]];
		for(j=0;j<data->features;j++) params.negcentroids[i][j] /= (observations-freq[data->Y[i]]);	
	}
	
	for(i=0;i<params.K;i++) 
	{
		normalizeVector(params.centroids[i],data->features);
		normalizeVector(params.negcentroids[i],data->features);
	}

	if(freq) free(freq);
	return params;
}

void writeSolutionToFile(char *fname,struct Data *data,struct Results *results)
{
	int i=0;
	FILE* fp = fopen(fname,"w");
	if(!fp) return;

	int ptr = 0;
	for(i=0;i<data->observations;i++)
	{
		if(data->mask[i]==0)
		{
			fprintf(fp, "%d %s\n", i+1,data->class_label[results->best_label[ptr]]);
			ptr = ptr + 1;
		}
	}
	fclose(fp);
}

void sortInputFile(char *fname)
{
	char buf[256];
	sprintf(buf,"sort -n -k1 -k2 %s -o %s",fname,fname);
	system(buf);
}

int getAccuracy(struct Data *data, struct Results *results)
{
	int correct = 0;
	int i=0;
	int pos = 0;
	for(i=0;i<data->observations;i++)
	{
		if(data->mask[i]==0)
			if(data->Y[i] == results->best_label[i]) correct = correct + 1;
	}
	return correct;
}

int main(int argc,char **argv)
{
	int i=0;
	int j=0;
	float p=0.0;
	float r=0.0;
	float f1=0.0;
	float f1max=0.0;
	float f1sum=0.0;

	double t1,t2,t3,t4,t5,t6;
    struct timeval tv1,tv2,tv3,tv4,tv5,tv6;
    char* measures = "measures.csv";


	if(9!=argc) puts("Wrong Input Format !!!");
	else
	{
		gettimeofday (&tv1, NULL);
		t1 = (double) (tv1.tv_sec) + 0.000001 * tv1.tv_usec;

		sortInputFile(argv[1]);
		struct Data data = readInput(argv[1]);
		struct rlabel *rlabel = readrlabel(argv[2],data.observations);
		readTrainFile(&data,argv[3]);
		readTestFile(&data,argv[4]);
		readClassFile(&data,argv[5]);
		readFeaturelabel(&data,argv[6]);
		
		featureTransformation(&data,argv[7]);    
        Normalize(&data,1); 
		Normalize(&data,0);

		gettimeofday (&tv2, NULL);
		t2 = (double) (tv2.tv_sec) + 0.000001 * tv2.tv_usec;

		gettimeofday (&tv3, NULL);
		t3 = (double) (tv3.tv_sec) + 0.000001 * tv3.tv_usec;

		struct CENTROIDParameters params = CENTROIDClassifier(&data);
		struct Results results = makePredictionsCENTROID(&data,&params);
		writeSolutionToFile(argv[8],&data,&results);
		for(i=0;i<data.totalClasses;i++)
			printf("Class %d and F1 Score %.6f\n", i+1,results.f1_score[i]);
		
		printf("Correctly Classified %d\n",getAccuracy(&data,&results));
		gettimeofday (&tv4, NULL);
		t4 = (double) (tv4.tv_sec) + 0.000001 * tv4.tv_usec;

		gettimeofday (&tv5, NULL);
		t5 = (double) (tv5.tv_sec) + 0.000001 * tv5.tv_usec;

		if(results.f1_score) free(results.f1_score);
		if(results.best_label) free(results.best_label);
		if(results.best_score) free(results.best_score);
		
		for(i=0;i<params.K;i++) 
		{
			if(params.centroids[i]) free(params.centroids[i]);
			if(params.centroids[i]) free(params.negcentroids[i]);
		}
		if(params.centroids) free(params.centroids);
		if(params.negcentroids) free(params.negcentroids);
		//outputfile argv[8];

	    if(data.csr.document) free(data.csr.document);
        if(data.csr.featureid) free(data.csr.featureid);
        if(data.csr.freq) free(data.csr.freq);
	

		if(rlabel) free(rlabel);
		if(data.mask) free(data.mask);
		if(data.Y) free(data.Y);
		if(data.feature_label)
		{
			for(i=0;i<data.features;i++) if(data.feature_label[i]) free(data.feature_label[i]);
			free(data.feature_label);
		}
		
		if(data.class_label)
		{
			for(i=0;i<data.observations;i++) if(data.class_label[i]) free(data.class_label[i]);
			free(data.class_label);
		}

		gettimeofday (&tv6, NULL);
		t6 = (double) (tv6.tv_sec) + 0.000001 * tv6.tv_usec;

		printf("\n\n#########  Time to read and process input %.6f seconds ###########\n",t2-t1);
		printf("######### Time to build Classifer and run on test set %.6f seconds ###########\n",t4-t3);
		printf("######### Time to free up memory resouces %.6f seconds ###########\n",t6-t5);

	}
	return 0;
}
