#include<stdio.h>
#include<string.h>
#include<stdlib.h>
#include<sys/time.h>
#include<math.h>
#include<assert.h>
#define max(a,b) a>b?a:b
#define min(a,b) a>b?b:a
#define SWAP(x,y) do { typeof(x) SWAP = x;x=y;y=SWAP;} while(0)

struct sparseRow
{
	int *document;
	int *featureid;
	float *freq;
	int nonZeros;
};

struct sparseCol
{
	int *document;
	int *featureid;
	float *freq;
	int nonZeros;
};

struct RidgeParameters
{
	float ***theta1;
	float *lambda;
	int *optimal;
	int paramlength;
};

struct Data
{
	struct sparseRow csr;	
	struct sparseCol csc;

	int *Y;
	int *mask;
	int *fmask;
	char **feature_label;
	char **class_label;
	
	int observations;
	int features;
	int totalClasses;
};

struct rlabel
{
	int docid;
	int fileid;
	char class_name[32];
};

struct WordPair
{
	int id;
	float weight;
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

void merge(struct WordPair *distances,struct WordPair *l,int lc,struct WordPair *r,int rc)
{
	int i=0;
	int j=0;
	int k=0;

	i=0;
	j=0;
	k=0;

	while(i<lc && j < rc)
	{
		if(l[i].weight < r[j].weight)
		{
			distances[k].weight = l[i].weight;
			distances[k].id = l[i].id;
			i++;
		}
		else
		{
			distances[k].weight = r[j].weight;
			distances[k].id = r[j].id;
			j++;
		}
		k++;
	}
	while(i<lc)
	{
		distances[k].weight = l[i].weight;
		distances[k].id = l[i].id;
		i++;
		k++;
	}
		
	while(j<rc)
	{
		distances[k].weight = r[j].weight;
		distances[k].id = r[j].id;
		j++;
		k++;
	}
}

void mergeSort(struct WordPair *distances,int n)
{
	if(n<2) return;	
	int m = n/2;
	int i=0;
	
	struct WordPair *l;
	struct WordPair *r;

	l = (struct WordPair *)calloc(m,sizeof(struct WordPair));
	r = (struct WordPair *)calloc(n-m,sizeof(struct WordPair));

	for(i=0;i<m;i++) 
	{
		l[i].weight = distances[i].weight;
		l[i].id = distances[i].id;
	}	
	
	for(i=m;i<n;i++)
	{
		r[i-m].weight = distances[i].weight;
		r[i-m].id = distances[i].id;
	}

	mergeSort(l,m);
	mergeSort(r,n-m);
	merge(distances,l,m,r,n-m);
	free(l);
	free(r);
}

void sort(struct WordPair *distances,int n,int order)
{
	mergeSort(distances,n);
	if(1==order)
	{
		int i=0;
		for(i=0;i<n/2;i++)
		{
			SWAP(distances[i].weight,distances[n-i-1].weight);
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
            printf("(%d %.6f) ",data->csr.featureid[j],data->csr.freq[j]);
        puts("");
    }

    puts("\n\n-------------\n\n");

    for(i=0;i<data->features;i++)
    {
    	for(j=data->csc.featureid[i];j<data->csc.featureid[i+1];j++) printf("(%d %.6f) ",data->csc.document[j],data->csc.freq[j]);
    	puts("");
    }
    puts("");
}

void createCSCMatrix(struct Data *data)
{
	int i=0;
	int j=0;
	
	data->csc.document = (int *)calloc(data->csr.nonZeros+1,sizeof(int));
	data->csc.featureid = (int *)calloc(data->features+1,sizeof(int));
	data->csc.freq = (float *)calloc(data->csr.nonZeros+1,sizeof(float));
	
	int *colptr = (int *)calloc(data->features+1,sizeof(int));
	for(i=0;i<data->observations;i++)
	{
		if(data->mask[i]==1)
		{
			for(j=data->csr.document[i];j<data->csr.document[i+1];j++)
				colptr[data->csr.featureid[j]] = colptr[data->csr.featureid[j]]+1;
		}
	}
	
	int csum=0;
	for(i=0;i<data->features;i++)
	{
		data->csc.featureid[i] = csum;
		csum = csum + colptr[i];
		colptr[i] = 0;
	}
	data->csc.featureid[data->features] = csum;

	for(i=0;i<data->features;i++) 
		colptr[i] = data->csc.featureid[i];

	for(i=0;i<data->observations;i++)
	{
		if(data->mask[i]==1)
		{
			for(j=data->csr.document[i];j<data->csr.document[i+1];j++)
			{
				data->csc.document[colptr[data->csr.featureid[j]]] = i;
				data->csc.freq[colptr[data->csr.featureid[j]]] = data->csr.freq[j];
				colptr[data->csr.featureid[j]] = colptr[data->csr.featureid[j]]+1;
			}
		}
	}
}

struct Data readInput(char *fname)
{
	struct Data data;
	FILE* fp=fopen(fname,"r");
	if(!fp) return data;

	data.observations = data.features = data.totalClasses = 0;
	data.Y = data.mask =  NULL;
	data.feature_label = NULL;
	data.class_label = NULL;
	data.fmask = NULL;

	data.csr.document = data.csr.featureid = NULL;
	data.csr.freq = NULL;

	data.csc.document = data.csc.featureid = NULL;
	data.csc.freq = NULL;	

	int documentid=0;
	int featureid=0;
	float freq=0.0;
	int i=0;
	int j=0;
	const int NSIZE = 64;
	int nonzeros = 0;
	int previous_document = -1;
	int rowptr = 0;
	int rowind = 0;
	int colptr = 0;
	int colind = 0;

	while(fscanf(fp,"%d%d%f",&documentid,&featureid,&freq)!=EOF)
	{
		data.observations  = max(data.observations,documentid);
		data.features = max(data.features,featureid);		
		nonzeros = nonzeros+1;
	}	
	fclose(fp);
	
	data.Y = (int *)calloc(data.observations+2,sizeof(int));
	data.mask = (int *)calloc(data.observations+2,sizeof(int));
	data.csr.nonZeros = nonzeros;	

	data.csr.document = (int *)calloc(data.observations+2,sizeof(int));
	data.csr.featureid = (int *)calloc(nonzeros+2,sizeof(int));
	data.csr.freq = (float *)calloc(nonzeros+2,sizeof(float));
	data.feature_label = (char**)calloc(data.features+1,sizeof(char*));
	data.class_label = (char**)calloc(data.observations+1,sizeof(char*));
	data.fmask = (int*)calloc(data.features+1,sizeof(int));

	for(i=0;i<data.observations;i++) data.class_label[i] = (char*)calloc(NSIZE,sizeof(char*));
    for(i=0;i<data.features;i++) data.feature_label[i] = (char *)calloc(NSIZE,sizeof(char*));

	fp = fopen(fname,"r");
	if(!fp) return data;

	while(fscanf(fp,"%d%d%f",&documentid,&featureid,&freq)!=EOF)
	{
		if(previous_document != documentid)
		{
			data.csr.document[rowptr] = colptr;
			rowptr = rowptr+1;
		}
		
		data.csr.featureid[colptr] = featureid-1;
		data.csr.freq[colptr] = freq;
		colptr=colptr+1;

		previous_document = documentid;
	}
	data.csr.document[rowptr] = colptr;
	return data;
}


float dotProduct(struct Data *data,float *weights,int docid)
{
	int j=0;
	float f = 0.0;
	for(j=data->csr.document[docid];j<data->csr.document[docid+1];j++)
			f = f + data->csr.freq[j] * weights[data->csr.featureid[j]];
	return f;
}

float cost(struct Data *data,struct RidgeParameters *params,int paramidx,int classid,int mask)
{
	float csum=0.0;
	float tsum=0.0;

	int i=0;
	int j=0;

	for(i=0;i<data->observations;i++)
	{
		if(data->mask[i]==mask)
		{
			tsum=dotProduct(data,params->theta1[paramidx][classid],i);
			tsum = tsum - data->Y[i];
			csum = csum + tsum*tsum;
		}
	}

	tsum = 0.0;
	for(i=0;i<data->features;i++) 
	{
		if(data->fmask[i]==1)
			tsum = tsum + (params->theta1[paramidx][classid][i]*params->theta1[paramidx][classid][i]);
	}
	tsum = tsum * params->lambda[paramidx];
	return csum + tsum;
}

float gradient(struct Data *data,float *dotProduct,struct RidgeParameters *params,int paramidx,int classid,int featureid)
{
	float tsum=0.0;
	float res=0.0;
	float sum=0.0;

	int i=0;
	int j=0;
	
	for(i=data->csc.featureid[featureid];i<data->csc.featureid[featureid+1];i++)
	{
		if(data->mask[data->csc.document[i]]==1)
		{
			tsum = (dotProduct[data->csc.document[i]] - data->csc.freq[i] * params->theta1[paramidx][classid][featureid]);
			tsum = data->Y[data->csc.document[i]] - tsum;
			res = res + tsum*data->csc.freq[i];
			sum = sum + (data->csc.freq[i])*(data->csc.freq[i]);
		}
	}

	return res/(sum+(float)params->lambda[paramidx]);
}

struct rlabel* readrlabel(char *fname,int n)
{
	FILE *fp = fopen(fname,"r");
	if(!fp) return NULL;

	int i=0;
	struct rlabel *data = (struct rlabel*)calloc(n+1,sizeof(struct rlabel));
	while(fscanf(fp,"%d%s%d",&data[i].docid,data[i].class_name,&data[i].fileid)!=EOF) 
	{
		data[i].docid = data[i].docid-1; //Everything is 0 based
		i++;
	}
	fclose(fp);
	return data;
}

void readTrainFile(struct Data *data,char *fname)
{
	int j=0;
	int docid=0;
	FILE *fp = fopen(fname,"r");
	if(!fp) return;
	while(fscanf(fp,"%d",&docid)!=EOF) 
	{
		data->mask[docid-1] = 1;
		for(j=data->csr.document[docid-1];j<data->csr.document[docid];j++) 
			data->fmask[data->csr.featureid[j]] = 1;
	}
	fclose(fp);
}

void readValidationFile(struct Data *data,char *fname)
{
	int docid=0;
	FILE *fp = fopen(fname,"r");
	if(!fp) return;
	while(fscanf(fp,"%d",&docid)!=EOF) data->mask[docid-1] = 2;
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
    	    for(j=data->csr.document[i];j<data->csr.document[i+1];j++)
    	    	csum = csum + data->csr.freq[j]*data->csr.freq[j];
    	    csum = sqrt(csum);
    		for(j=data->csr.document[i];j<data->csr.document[i+1];j++) 
    			data->csr.freq[j] /= csum;
	    }
    }
}

void writeSolutionToFile(char *fname,struct Data *data,struct Predictions *results)
{
	int i=0;
	FILE* fp = fopen(fname,"w");
	if(!fp) return;

	int ptr = 0;
	for(i=0;i<data->observations;i++)
	{
		if(data->mask[i]==0)
		{
			fprintf(fp, "%d %s\n", i+1,data->class_label[results[ptr].label]);
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

float *getDotProduct(struct Data *data,struct RidgeParameters *params,int paramidx,int classidx)
{
	int i=0;
	int j=0;
	
	float *dotProd = (float *)calloc(data->observations+1,sizeof(float));
	for(i=0;i<data->observations;i++)
	{
		if(data->mask[i]==1)
			dotProd[i] = dotProduct(data,params->theta1[paramidx][classidx],i);
	}
	return dotProd;
}

float getF1Score(struct Predictions *pred,int observations,int iter)
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
	}
	return best_f1;
}

void initResults(struct Results *results,int totalClasses,int observations,int test_observations)
{
	results->f1_score = (float *)calloc(totalClasses,sizeof(float));
	results->best_label = (int *)calloc(observations+1,sizeof(int));
	results->best_score = (float *)calloc(observations+1,sizeof(float));
	results->test_observations = test_observations;
}

void validate(struct Data *data,struct RidgeParameters *params)
{
	int i=0,j=0,k=0,p=0,ptr=0,validation_observations=0;
	float f=0.0,tsum=0.0,csum=0.0;
	
	for(i=0;i<data->observations;i++) if(data->mask[i]==2) validation_observations = validation_observations + 1;
		
	struct Predictions *pred;
	pred = (struct Predictions *)calloc(validation_observations,sizeof(struct Predictions));
	float *validation_scores = (float*)calloc(data->totalClasses,sizeof(float));

	for(p=0;p<params->paramlength;p++)
	{
		for(k=0;k<data->totalClasses;k++)
		{
			ptr=0;
			for(i=0;i<data->observations;i++)
			{
				if(data->mask[i]==2)
				{
					pred[ptr].id = i;
					pred[ptr].score = dotProduct(data,params->theta1[p][k],i);
					pred[ptr].label = data->Y[i];
					ptr = ptr + 1;
				}
			}
			tsort(pred,ptr,1);
			f = getF1Score(pred,ptr,k);

			if(f > validation_scores[k])
			{
				validation_scores[k] = f;
				params->optimal[k] = p;
			}
		}
	}

	if(validation_scores) free(validation_scores);
}

struct Predictions* makePredictionsRidge(struct Data *data,struct RidgeParameters *params)
{
	int i=0,j=0,k=0,label=0,ptr=0;
	float f = 0.0,best_f=0.0;
	
	struct Predictions *pred;
	pred = (struct Predictions *)calloc(data->observations,sizeof(struct Predictions));

	for(i=0;i<data->observations;i++)
	{
		if(data->mask[i]==0)
		{
			best_f = 0.0;
			for(k=0;k<data->totalClasses;k++)
			{
				f = dotProduct(data,params->theta1[params->optimal[k]][k],i);
				if(f>best_f)
				{
					best_f = f;
					label = k;
				}
			}
			pred[ptr].id = i;
			pred[ptr].score = best_f;
			pred[ptr].label = label;
			ptr = ptr+1;
		}
	}
	return pred;
}

struct RidgeParameters RidgeClassifier(struct Data *data)
{
	int i=0;
	int j=0;
	int k=0;
	int p=0;
	int z=0;

	float c1=0.0;
	float c2=0.0;
	const int iter = 100;
	float nweight =0.0;
	int totalParams = 6;
	int validation_score = 0;
	int best_validation_score = 0;

	struct RidgeParameters params;
	params.lambda = (float *)calloc(totalParams,sizeof(float));	
	params.optimal = (int*)calloc(data->totalClasses,sizeof(int));
	params.paramlength= 6;

	params.lambda[0] = 0.01;	
	params.lambda[1] = 0.05;	
	params.lambda[2] = 0.1;	
	params.lambda[3] = 0.5;	
	params.lambda[4] = 1;	
	params.lambda[5] = 10;	

	params.theta1 = (float ***)calloc(params.paramlength,sizeof(float**));
	for(i=0;i<params.paramlength;i++) 
	{
		params.theta1[i] = (float **)calloc(data->totalClasses,sizeof(float *));
		for(j=0;j<data->totalClasses;j++) 
		{
			params.theta1[i][j] = (float*)calloc(data->features+1,sizeof(float));
		}
	}

	int *tlabels;//temporary array to keep of Y-values(class labels in original data)
	tlabels = (int*)calloc(data->observations,sizeof(int));
	for(i=0;i<data->observations;i++) tlabels[i] = data->Y[i];
	
	for(k=0;k<params.paramlength;k++)
	{
		for(p=0;p<data->totalClasses;p++)
		{
			for(i=0;i<data->observations;i++)
			{
				if(data->mask[i]==1)
				{
					if(data->Y[i]!=p) data->Y[i]=0;
					else data->Y[i] = 1;
				}
			}

			for(i=0;i<iter;i++)
			{
				c1 = cost(data,&params,k,p,1);
				float *dp = getDotProduct(data,&params,k,p);
				for(j=0;j<data->features;j++) 
				{
					if(0 == data->fmask[j]) continue;
					nweight = max(0,gradient(data,dp,&params,k,p,j));
					for(z=data->csc.featureid[j];z<data->csc.featureid[j+1];z++)
						dp[data->csc.document[z]] = dp[data->csc.document[z]] + data->csc.freq[z]*(nweight - params.theta1[k][p][j]);
					params.theta1[k][p][j] = nweight;
				}
				c2 = cost(data,&params,k,p,1);
				free(dp);
				if(fabs(c1-c2) < 1e-6) break;
 			}
 			for(i=0;i<data->observations;i++) data->Y[i] = tlabels[i];
		}
	}
	validate(data,&params);
	free(tlabels);
	return params;
}

int getAccuracy(struct Data *data, struct Predictions *results)
{
	int correct = 0;
	int i=0;
	int ptr=0;
	for(i=0;i<data->observations;i++)
	{
		if(data->mask[i]==0)
		{
			if(data->Y[i] == results[i].label)
				correct = correct + 1;
			ptr = ptr+1;
		}
	}
	return correct;
}

int **confustionMatrix(struct Data *data,struct Predictions *results)
{
    int i=0;
    int j=0;
    int **mat;

    mat = (int **)calloc(data->totalClasses,sizeof(int*));
    for(i=0;i<data->totalClasses;i++) mat[i] = (int *)calloc(data->totalClasses,sizeof(int));

    for(i=0;i<data->observations;i++)
    {
    	if(data->mask[i]==0)
    	{
    		mat[results[j].label][data->Y[i]] = mat[results[j].label][data->Y[i]] + 1;
    		j = j + 1;
    	}
    }
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

int main(int argc,char **argv)
{
	int i=0;
	int j=0;
	float p=0.0;
	float r=0.0;
	float f1=0.0;
	float f1max=0.0;
	const int NSIZE = 64;
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
		readValidationFile(&data,argv[9]);
		readTestFile(&data,argv[4]);
		readClassFile(&data,argv[5]);
		readFeaturelabel(&data,argv[6]);
		
	    featureTransformation(&data,argv[7]);    
        Normalize(&data,1);
        Normalize(&data,2); 
		Normalize(&data,0);
		createCSCMatrix(&data);
		
		gettimeofday (&tv2, NULL);
		t2 = (double) (tv2.tv_sec) + 0.000001 * tv2.tv_usec;

		gettimeofday (&tv3, NULL);
		t3 = (double) (tv3.tv_sec) + 0.000001 * tv3.tv_usec;		

		struct RidgeParameters params = RidgeClassifier(&data);
		struct Predictions* results = makePredictionsRidge(&data,&params);
		writeSolutionToFile(argv[8],&data,results);

		int ** confmat = confustionMatrix(&data,results);
		for(i=0;i<data.totalClasses;i++)
		{
			p = precision(confmat,i,data.totalClasses);
			r = recall(confmat,i,data.totalClasses);
			f1 = (p+r==0?0:2*p*r/(p+r));
			f1max = max(f1max,f1);
			f1sum = f1sum + f1;
			printf("Class %d Optimal lambda %.6f and F1 Score %.6f\n", i+1,params.lambda[params.optimal[i]],f1);
		}

		char *top_words_fname = (char *)calloc(NSIZE,sizeof(char));
		strcpy(top_words_fname,strstr(argv[1],"word") ? "top_features_word_" : "top_features_word_char_");
		strcat(top_words_fname,argv[7]);
		strcat(top_words_fname,".txt");
		FILE *top_features = fopen(top_words_fname,"w");
		for(i=0;i<data.totalClasses;i++)
		{
			struct WordPair *wp = (struct WordPair*)calloc(data.features+1,sizeof(struct WordPair));
			for(j=0;j<data.features;j++)
			{
				wp[j].id = j;
				wp[j].weight = params.theta1[params.optimal[i]][i][j];
			}
			sort(wp,data.features,1);
			for(j=0;j<10;j++) fprintf(top_features,"%d,%s,%.6f\n",i+1,data.feature_label[wp[j].id],wp[j].weight);
			fprintf(top_features, "\n");
			free(wp);
		}

		fclose(top_features);
		
		gettimeofday (&tv4, NULL);
		t4 = (double) (tv4.tv_sec) + 0.000001 * tv4.tv_usec;

		gettimeofday (&tv5, NULL);
		t5 = (double) (tv5.tv_sec) + 0.000001 * tv5.tv_usec;

		if(params.theta1)
        {
        	for(i=0;i<params.paramlength;i++)
        	{
        		for(j=0;j<data.totalClasses;j++)
        		{
        			if(params.theta1[i][j]) free(params.theta1[i][j]);
        		}
        		if(params.theta1[i]) free(params.theta1[i]);
        	}
        	free(params.theta1);
        }

        if(params.lambda) free(params.lambda);
		if(data.csr.document) free(data.csr.document);
        if(data.csr.featureid) free(data.csr.featureid);
        if(data.csr.freq) free(data.csr.freq);

        if(data.csc.document) free(data.csc.document);
        if(data.csc.featureid) free(data.csc.featureid);
        if(data.csc.freq) free(data.csc.freq);        

        if(data.fmask) free(data.fmask);
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

		for(i=0;i<data.totalClasses;i++) if(confmat[i]) free(confmat[i]);
		free(confmat);
	
		gettimeofday (&tv6, NULL);
		t6 = (double) (tv6.tv_sec) + 0.000001 * tv6.tv_usec;

		printf("\n\n#########  Time to read and process input %.6f seconds ###########\n",t2-t1);
		printf("######### Time to build Classifer and run on test set %.6f seconds ###########\n",t4-t3);
		printf("######### Time to free up memory resouces %.6f seconds ###########\n",t6-t5);
		printf("Top weighted Features written to file %s\n space sepearated",top_words_fname);
	}
	return 0;
}
