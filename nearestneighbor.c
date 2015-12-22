#include<stdio.h>
#include<string.h>
#include<stdlib.h>
#include<math.h>
#include<assert.h>
#include<sys/time.h>
#define max(a,b) a>b?a:b
#define min(a,b) a>b?b:a
#define SWAP(x,y) do { typeof(x) SWAP = x;x=y;y=SWAP;} while(0)

struct SparseValue
{
	int featureid;
	float freq;
};

struct SparseData
{
	int *document;
	struct SparseValue *spv;
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

struct KNNDistance
{
	int id;
	float distance;
};

struct KNNParameters
{
	int K;
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

void tmerge(struct Predictions *distances,struct Predictions *l,int lc,struct Predictions *r,int rc)
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

void tmergeSort(struct Predictions *distances,int n)
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

	tmergeSort(l,m);
	tmergeSort(r,n-m);
	tmerge(distances,l,m,r,n-m);
	free(l);
	free(r);
}

void tsort(struct Predictions *distances,int n,int order)
{
	tmergeSort(distances,n);
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

void merge(struct KNNDistance *distances,struct KNNDistance *l,int lc,struct KNNDistance *r,int rc)
{
	int i=0;
	int j=0;
	int k=0;

	i=0;
	j=0;
	k=0;

	while(i<lc && j < rc)
	{
		if(l[i].distance < r[j].distance)
		{
			distances[k].distance = l[i].distance;
			distances[k].id = l[i].id;
			i++;
		}
		else
		{
			distances[k].distance = r[j].distance;
			distances[k].id = r[j].id;
			j++;
		}
		k++;
	}
	while(i<lc)
	{
		distances[k].distance = l[i].distance;
		distances[k].id = l[i].id;
		i++;
		k++;
	}
		
	while(j<rc)
	{
		distances[k].distance = r[j].distance;
		distances[k].id = r[j].id;
		j++;
		k++;
	}
}

void mergeSort(struct KNNDistance *distances,int n)
{
	if(n<2) return;	
	int m = n/2;
	int i=0;
	
	struct KNNDistance *l;
	struct KNNDistance *r;

	l = (struct KNNDistance *)calloc(m,sizeof(struct KNNDistance));
	r = (struct KNNDistance *)calloc(n-m,sizeof(struct KNNDistance));

	for(i=0;i<m;i++) 
	{
		l[i].distance = distances[i].distance;
		l[i].id = distances[i].id;
	}	
	
	for(i=m;i<n;i++)
	{
		r[i-m].distance = distances[i].distance;
		r[i-m].id = distances[i].id;
	}

	mergeSort(l,m);
	mergeSort(r,n-m);
	merge(distances,l,m,r,n-m);
	free(l);
	free(r);
}

void sort(struct KNNDistance *distances,int n,int order)
{
	mergeSort(distances,n);
	if(1==order)
	{
		int i=0;
		for(i=0;i<n/2;i++)
		{
			SWAP(distances[i].distance,distances[n-i-1].distance);
			SWAP(distances[i].id,distances[n-i-1].id);
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
            printf("(%d %.6f) ",data->csr.spv[j].featureid,data->csr.spv[j].freq);
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
	data.csr.spv = (struct SparseValue *)calloc(nonzeros+1,sizeof(struct SparseValue));
	
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
		data.csr.spv[colptr].featureid = featureid-1;
		data.csr.spv[colptr].freq = freq;
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
		if(data->mask[i]==1)
		{
			required_docs = required_docs+1;
			for(j=data->csr.document[i];j<data->csr.document[i+1];j++)
				feature[data->csr.spv[j].featureid] = feature[data->csr.spv[j].featureid] + 1;
		}
	}
	
	for(i=0;i<data->observations;i++)
	{
		for(j=data->csr.document[i];j<data->csr.document[i+1];j++)
		{
			if(1 == format) data->csr.spv[j].freq = 1.0;
			else if(2 == format) data->csr.spv[j].freq = sqrt(data->csr.spv[j].freq);
			else if(3 == format) data->csr.spv[j].freq = log((float)required_docs/feature[data->csr.spv[j].featureid]);
			else if(4 == format) data->csr.spv[j].freq = sqrt(data->csr.spv[j].freq) * log((float)required_docs/feature[data->csr.spv[j].featureid]);
			else if(5 == format) data->csr.spv[j].freq = data->csr.spv[j].freq * log((float)required_docs/feature[data->csr.spv[j].featureid]);
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
    	    for(j=data->csr.document[i];j<data->csr.document[i+1];j++) csum = csum + data->csr.spv[j].freq*data->csr.spv[j].freq;
    	    csum = sqrt(csum);
    	    for(j=data->csr.document[i];j<data->csr.document[i+1];j++) data->csr.spv[j].freq /= csum;
	    }
    }
}

float getDistance(struct Data *data,int p,int q)
{
	//type = 1 for csr
	float csum=0.0;
	int i=0;
    int j=0;
    int n1=0;
    int n2=0;

    i = data->csr.document[p];
    n1 = data->csr.document[p+1];
    
    i = data->csr.document[p];
    n1 = data->csr.document[p+1];

    j = data->csr.document[q];
    n2 = data->csr.document[q+1];

    while( i < n1 && j < n2 )
    {
    	if(data->csr.spv[i].featureid == data->csr.spv[j].featureid)
        {
        	csum = csum + data->csr.spv[i].freq * data->csr.spv[j].freq;
            i++;
            j++;
        } 
        else if( data->csr.spv[i].featureid > data->csr.spv[j].featureid ) j++;
        else i++;
    } 
    return csum;
}

int getMajorityLabel(struct Data *data,struct KNNDistance *distances,int n,struct KNNParameters params,int idx)
{
	const int NSIZE = 100; //i.e class labels can be from 0-99
	int i=0;
	int j=0;
	int *freq;

	float pos = 0.0;
	float neg = 0.0;
	
	int label = 0;
	float best = -1;
	freq = (int *)calloc(NSIZE,sizeof(int));

	for(i=0;i<params.K;i++)
		freq[data->Y[distances[i].id]] = freq[data->Y[distances[i].id]]+1;
	
	for(i=0;i<data->totalClasses;i++)
	{
		if(freq[i]==0) continue;
		//ith class is postive and everything else is negative
		pos = 0.0;
		neg = 0.0;	
		for(j=0;j<params.K;j++)
		{
			if(data->Y[distances[j].id]==i) pos = pos + getDistance(data,distances[j].id,idx);
			else neg = neg + getDistance(data,distances[j].id,idx);
		}
		if(pos-neg > best)
		{
			best = pos-neg;
			label = i;
		}
	}

	free(freq);
	return label;
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

void populateNearestNeighbor(struct Data *data,struct Predictions **tpred)
{
	int i=0,j=0,tptr=0,ptr=0;
	float sim=0.0;

	for(i=0;i<data->observations;i++)	
	{
		if(data->mask[i]==0)
		{
			tptr = 0;
			for(j=0;j<data->observations;j++)
			{
				if(data->mask[j]==1)
				{
					sim = getDistance(data,i,j);
					tpred[ptr][tptr].id = i;
					tpred[ptr][tptr].score = sim;
					tpred[ptr][tptr].label = data->Y[j];
					tptr = tptr + 1;
				}
			}
			tsort(tpred[ptr],tptr,1);
			ptr = ptr + 1;
		}
	}
}

struct Results makePredictionsKNN(struct Data *data,struct KNNParameters *params)
{
	int i=0,j=0,k=0,label=0,ptr=0,tptr=0,tp=0,fp=0,fn=0,test_observations=0;
	float sim=0.0,f1=0.0,prec=0.0,recall=0.0,best_f1=0.0;

	for(i=0;i<data->observations;i++) 
		if(data->mask[i]==0) 
			test_observations = test_observations + 1;

	struct Predictions *pred = (struct Predictions *)calloc(test_observations,sizeof(struct Predictions));
	struct Predictions **tpred = (struct Predictions **)calloc(test_observations,sizeof(struct Predictions*));
	for(i=0;i<test_observations;i++) 
		tpred[i] = (struct Predictions *)calloc(data->observations+1,sizeof(struct Predictions));

	populateNearestNeighbor(data,tpred);

	struct Results results;
	initResults(&results,data->totalClasses,data->observations,test_observations);

	for(k=0;k<data->observations;k++) results.best_score[k] = -1e7;

	for(k=0;k<data->totalClasses;k++)
	{
		ptr = 0;
		for(i=0;i<data->observations;i++)
		{
			if(data->mask[i]==0)
			{
				sim = 0.0;
				for(j=0;j<params->K;j++)
				{
					if(tpred[ptr][j].label == k) sim = sim + tpred[ptr][j].score;
					else sim = sim - tpred[ptr][j].score;
				}

				pred[ptr].id = i;
				pred[ptr].score = sim;
				pred[ptr].label = data->Y[i];
				ptr = ptr + 1;
			}
		}

		tsort(pred,ptr,1);
		setF1Score(&results,pred,ptr,k);
	}

	for(i=0;i<test_observations;i++) if(tpred[i]) free(tpred[i]);
	if(tpred) free(tpred);
	
	if(pred) free(pred);
	return results;
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

int **confustionMatrix(int *obtained,int *correct,int n,int totalClasses)
{
    int i=0;
    int j=0;
    int **mat;

    mat = (int **)calloc(totalClasses,sizeof(int*));
    for(i=0;i<totalClasses;i++) mat[i] = (int *)calloc(totalClasses,sizeof(int));
    for(i=0;i<n;i++) mat[obtained[i]][correct[i]] = mat[obtained[i]][correct[i]]+1;
    return mat;
}

float precision(int **confmat,int label,int n)
{
	int i=0;
	int den=0;
	int num=confmat[label][label];
	for(i=0;i<n;i++) den = den + confmat[label][i];
	return den==0?0:(float)num/den;
}

float recall(int **confmat,int label,int n)
{
	int i=0;
	int den=0;
	int num=confmat[label][label];
	for(i=0;i<n;i++) den = den + confmat[i][label];
	return den==0?0:(float)num/den;
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
	float f1max=0.0;
	float p=0.0;
	float r=0.0;
	float f1=0.0;
	float f1sum=0.0;

	double t1,t2,t3,t4,t5,t6;
    struct timeval tv1,tv2,tv3,tv4,tv5,tv6;
    char* measures = "measures.csv";

	if(10!=argc) puts("Wrong Input Format !!!");
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

		struct KNNParameters params;
		params.K = atoi(argv[9]);
        struct Results results = makePredictionsKNN(&data,&params);
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

		if(data.csr.document) free(data.csr.document);
	    if(data.csr.spv) free(data.csr.spv);

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
