import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from wordcloud import WordCloud,STOPWORDS
import pickle
import pydeck as pdk
import re
from collections import Counter
from PIL import Image

#import variables

#########################  a faire #########################################
# 
#
###########################################################################"


st.set_page_config(layout="wide")


#import des données
@st.cache
def load_data():
	coopi = pd.read_csv('viz_coopi.csv',sep='\t')
	coopi.drop([i for i in coopi if 'Unnamed' in i],axis=1,inplace=True)
	correl_coopi=pd.read_csv('graphs_coopi.csv',sep='\t')
	questions_coopi=pd.read_csv('questions_coopi.csv',sep='\t')
	questions_coopi.drop([i for i in questions_coopi.columns if 'Unnamed' in i],axis=1,inplace=True)
	quest_coopi=questions_coopi.iloc[3].to_dict()
	questions_coopi=questions_coopi.T
	sankey_coopi=questions_coopi[questions_coopi[1]=='sankey'].index.tolist()
	codes_coopi=pd.read_csv('codes_coopi.csv',index_col=None,sep='\t').dropna(how='any',subset=['color'])
	
	avsi = pd.read_csv('viz_avsi.csv',sep='\t')
	avsi.drop([i for i in avsi if 'Unnamed' in i],axis=1,inplace=True)
	avsi['E6_Boreholes_village']=avsi['E6_Boreholes_village'].apply(lambda x:str(x))
	correl_avsi=pd.read_csv('graphs_avsi.csv',sep='\t')
	questions_avsi=pd.read_csv('questions_avsi.csv',sep='\t')
	questions_avsi.drop([i for i in questions_avsi.columns if 'Unnamed' in i],axis=1,inplace=True)
	quest_avsi=questions_avsi.iloc[3].to_dict()
	questions_avsi=questions_avsi.T
	sankey_avsi=questions_avsi[questions_avsi[1]=='sankey'].index.tolist()
	codes_avsi=pd.read_csv('codes_avsi.csv',index_col=None,sep='\t').dropna(how='any',subset=['color'])
	
	return coopi,correl_coopi,quest_coopi,codes_coopi,sankey_coopi,avsi,correl_avsi,quest_avsi,codes_avsi,sankey_avsi

coopi,correl_coopi,questions_coopi,codes_coopi,sankey_coopi,avsi,correl_avsi,questions_avsi,codes_avsi,sankey_avsi=load_data()

#st.write(sankey)
#st.dataframe(correl)
#st.write(data.columns)
#st.write(correl.shape)

def sankey_graph(data,L,height=600,width=1600):
    """ sankey graph de data pour les catégories dans L dans l'ordre et 
    de hauter et longueur définie éventuellement"""
    
    nodes_colors=["blue","green","grey",'yellow',"coral",'darkviolet','saddlebrown','darkblue','brown']
    link_colors=["lightblue","limegreen","lightgrey","lightyellow","lightcoral",'plum','sandybrown','lightsteelblue','rosybrown']
    
    
    labels=[]
    source=[]
    target=[]
    
    for cat in L:
        lab=data[cat].unique().tolist()
        lab.sort()
        labels+=lab
    
    #st.write(labels)
    
    for i in range(len(data[L[0]].unique())): #j'itère sur mes premieres sources
    
        source+=[i for k in range(len(data[L[1]].unique()))] #j'envois sur ma catégorie 2
        index=len(data[L[0]].unique())
        target+=[k for k in range(index,len(data[L[1]].unique())+index)]
        
        for n in range(1,len(L)-1):
        
            source+=[index+k for k in range(len(data[L[n]].unique())) for j in range(len(data[L[n+1]].unique()))]
            index+=len(data[L[n]].unique())
            target+=[index+k for j in range(len(data[L[n]].unique())) for k in range(len(data[L[n+1]].unique()))]
       
    iteration=int(len(source)/len(data[L[0]].unique()))
    value_prov=[(int(i//iteration),source[i],target[i]) for i in range(len(source))]
    
    
    value=[]
    k=0
    position=[]
    for i in L:
        k+=len(data[i].unique())
        position.append(k)
    
   
    
    for triplet in value_prov:    
        k=0
        while triplet[1]>=position[k]:
            k+=1
        
        df=data[data[L[0]]==labels[triplet[0]]].copy()
        df=df[df[L[k]]==labels[triplet[1]]]
        #Je sélectionne ma première catégorie
        value.append(len(df[df[L[k+1]]==labels[triplet[2]]]))
        
    color_nodes=nodes_colors[:len(data[L[0]].unique())]+["black" for i in range(len(labels)-len(data[L[0]].unique()))]
    #st.write(color_nodes)
    color_links=[]
    for i in range(len(data[L[0]].unique())):
    	color_links+=[link_colors[i] for couleur in range(iteration)]
    #st.write(L,len(L),iteration)
    #st.write(color_links)
   
   
    fig = go.Figure(data=[go.Sankey(
    node = dict(
      pad = 15,
      thickness = 30,
      line = dict(color = "black", width = 1),
      label = [i.upper() for i in labels],
      color=color_nodes
      )
      
    ,
    link = dict(
      source = source, # indices correspond to labels, eg A1, A2, A1, B1, ...
      target = target,
      value = value,
      color = color_links))])
    return fig


def count2(abscisse,ordonnée,dataf,codes,legendtitle='',xaxis=''):
	
	dataf[ordonnée]=dataf[ordonnée].apply(lambda x:str(x))
	agg=dataf[[abscisse,ordonnée]].groupby(by=[abscisse,ordonnée]).aggregate({abscisse:'count'}).unstack().fillna(0)
	agg2=agg.T/agg.T.sum()
	agg2=agg2.T*100
	agg2=agg2.astype(int)
    
	if abscisse=='dam ':
		agg=agg.reindex(['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'])
		agg2=agg2.reindex(['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'])
    
	x=agg.index
    
	#st.write(agg)
	#st.write(agg2)
	
	if ordonnée.split(' ')[0] in codes['list name'].values:
		#st.write('on est là')
		colors_code=codes[codes['list name']==ordonnée.split(' ')[0]].sort_values(['coding'])
		labels=colors_code['label'].tolist()
		colors=colors_code['color'].tolist()
		fig = go.Figure()
		#st.write(labels,colors)
		#st.write(x)
		for i in range(len(labels)):
			if labels[i] in dataf[ordonnée].unique():
				#st.write(labels[i])
				#st.write(agg.columns)
				#st.write(agg[(abscisse,str(labels[i]))])
				fig.add_trace(go.Bar(x=x, y=agg[(abscisse,str(labels[i]))], name=str(labels[i]),\
                           marker_color=colors[i].lower(),customdata=agg2[(abscisse,str(labels[i]))],textposition="inside",\
                           texttemplate="%{customdata} %",textfont_color="black"))
        
	else:
		fig = go.Figure(go.Bar(x=x, y=agg.iloc[:,0], name=agg.columns.tolist()[0][1],marker_color='green',customdata=agg2.iloc[:,0],textposition="inside",\
                           texttemplate="%{customdata} %",textfont_color="black"))
		for i in range(len(agg.columns)-1):
			fig.add_trace(go.Bar(x=x, y=agg.iloc[:,i+1], name=agg.columns.tolist()[i+1][1],customdata=agg2.iloc[:,i+1],textposition="inside",\
                           texttemplate="%{customdata} %",textfont_color="black"))
    
	fig.update_layout(barmode='relative', \
                  xaxis={'title':xaxis,'title_font':{'size':18}},\
                  yaxis={'title':'Persons','title_font':{'size':18}})
	fig.update_layout(legend_title=legendtitle,legend=dict(orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1.01,font=dict(size=18),title=dict(font=dict(size=18))
    ))
    #fig.update_layout(title_text=title)
    
	return fig

def pourcent2(abscisse,ordonnée,dataf,codes,legendtitle='',xaxis=''):
    
	agg2=dataf[[abscisse,ordonnée]].groupby(by=[abscisse,ordonnée]).aggregate({abscisse:'count'}).unstack().fillna(0)
	agg=agg2.T/agg2.T.sum()
	agg=agg.T.round(2)*100
    
    
	if abscisse=='dam ':
		agg=agg.reindex(['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'])
		agg2=agg2.reindex(['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'])
    
	x=agg2.index
    
	if ordonnée.split(' ')[0] in codes['list name'].values:
		colors_code=codes[codes['list name']==ordonnée.split(' ')[0]].sort_values(['coding'])
		labels=colors_code['label'].tolist()
		colors=colors_code['color'].tolist()
		fig = go.Figure()
        
		for i in range(len(labels)):
			if labels[i] in dataf[ordonnée].unique():
				fig.add_trace(go.Bar(x=x, y=agg[(abscisse,labels[i])], name=labels[i],\
                           marker_color=colors[i].lower(),customdata=agg2[(abscisse,labels[i])],textposition="inside",\
                           texttemplate="%{customdata} persons",textfont_color="black"))
        
	else:
        #st.write(agg)
        #st.write(agg2)
		fig = go.Figure(go.Bar(x=x, y=agg.iloc[:,0], name=agg.columns.tolist()[0][1],marker_color='green',customdata=agg2.iloc[:,0],textposition="inside",\
                           texttemplate="%{customdata} persons",textfont_color="black"))
		for i in range(len(agg.columns)-1):
			fig.add_trace(go.Bar(x=x, y=agg.iloc[:,i+1], name=agg.columns.tolist()[i+1][1],customdata=agg2.iloc[:,i+1],textposition="inside",\
                           texttemplate="%{customdata} persons",textfont_color="black"))
    
	fig.update_layout(barmode='relative', \
                  xaxis={'title':xaxis,'title_font':{'size':18}},\
                  yaxis={'title':'Pourcentage','title_font':{'size':18}})
	fig.update_layout(legend_title=legendtitle,legend=dict(orientation='h',
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1.01,font=dict(size=18),title=dict(font=dict(size=18))
    ))
    #fig.update_layout(title_text=title)    
	return fig


img1 = Image.open("logoAxiom.png")
img2 = Image.open("logoCOOPI.png")
img3 = Image.open("logoAvsi.png")

def main():	
	
	
	
	#st.write(codes)
	
	st.sidebar.image(img1,width=200)
	st.sidebar.title("")
	st.sidebar.title("")
	database=st.sidebar.selectbox('',['Choose your database','COOPI','AVSI'])
	
	title1, title3 = st.columns([9,2])

	
	if database=='COOPI':
		topic = st.sidebar.radio('What do you want to do ?',('Display machine learning results','Display correlations',\
		'Display Sankey Graphs','Display Wordclouds'))
		title3.image(img2)
		data=coopi.copy()
		correl=correl_coopi.copy()
		questions=questions_coopi.copy()
		codes=codes_coopi.copy()
		sankey=sankey_coopi.copy()
		
	elif database=='AVSI':
		topic = st.sidebar.radio('What do you want to do ?',('Display correlations','Display Sankey Graphs','Display Wordclouds'))
		title3.image(img3)
		data=avsi.copy()
		correl=correl_avsi.copy()
		questions=questions_avsi.copy()
		codes=codes_avsi.copy()
		sankey=sankey_avsi.copy()
	else:
		topic=''
	
		
	
	if topic=='Display machine learning results':
		
		title1.title('Machine learning results on predictive model trained on : ')
		title1.title('Question: Can you say that your agricultural productivity has increased as result of the project')
		
		
		st.title('')
		st.markdown("""---""")	
		st.subheader('Note:')
		st.write('A machine learning model has been run on the question related to impression of adequate conditions in the settlement, the objective of this was to identify specificaly for these question which are the parameters that influence the most the wellbeing of beneficiaries. The models are run in order to try to predict as precisely as possible the feeling that the respondents expressed in their response to this question. The figures below shows which parameters have a greater impact in the prediction of the model than a normal random aspect (following a statistic normal law)')
		st.write('')
		st.write('Each line of the graph represents one feature of the survey that is important to predict the response to the question.')
		st.write('Each point on the right of the feature name represents one person of the survey. A red point represent a high value to the specific feature and a blue point a low value (a purple one a value inbetween).')
		st.write('SHAP value: When a point is on the right side, it means that it contributed to a better note while on the left side, this specific caracter of the person reduced the final result of the prediction. In this case it is 1 for adequate condition and 0 for not adequate so each point on the right tends to contibute to a better feeling.')
		st.write('')
		st.write('The coding for the responses is indicated under the graph and the interpretation of the graphs is written below.')
		st.markdown("""---""")	
				
		temp = Image.open('shap.png')
		image = Image.new("RGBA", temp.size, "WHITE") # Create a white rgba background
		image.paste(temp, (0, 0), temp)
		st.image(image, use_column_width = True)
		
		a, b = st.columns([1,1])
		a.caption('Did you receive any agricultural equipment from COOPI? Received 1 - Did not receive : 0')
		a.caption('Currently, are you receiving any of the following assistance from COOPI?   [Agricultural support] Receive Agricultural support : 1 - Does not receive : 0')
		a.caption('What crops did you grow on this plot? [Cowpea]  Grows Cowpea : 1 - Does not : 1')
		a.caption('Currently, are you receiving any of the following assistance from COOPI? [CFW] Is enroled in CFW : 1 - is not : 0')
		b.caption('What crops did you grow on this plot? [Maize]  Grows Maize : 1 - Does not : 1')
		b.caption('What crops did you grow on this plot? [Tomatoes]  Grows Tomatoes : 1 - Does not : 1')
		b.caption('Did you or any other member of your household (available) use a latrine/toilet the last time you defecated?: Yes:1 - No:0')
		
		st.write('We can see that the main parameter for feeling agricultural productivity has increased as result of the project are the fact to have received equipment and agricultural support from COOPI. This seems to show the effectiveness of these two activities')
		st.write('We see also that beeing a beneficiary of CFW activities also tends to increase the probability to fees that the productivity has increased howeve, we can see below that when people have to work a lot of days as CFW it seems to reduce the productivity. It might then be interesting to multiply the number of beneficiary in CFW and reduce the number of working days if the beneficiary are farmers.')
		st.write('Also we see that the increase on productivity is mainly for cowpeas, Maize and Tomatoes. And that when people have large surfaces they are more likely to have had the feeling that their productivity had increased.')
		st.write('It seems that those not having access to latrines are more likely to have had increased productivity (probably because they are more likely to be host outside of central village')
		
		
		st.markdown("""---""")	
	
	
	
				
		
	elif topic=='Display correlations':	
		
		title1.title('Main correlations uncovered from the database')
		
		
		if database=='COOPI':
			continues=pickle.load( open( "cont_feat_coopi.p", "rb" ) )
			cat_cols=pickle.load( open( "cat_cols_coopi.p", "rb" ) )
			st.write('Note: There are huge differences between Baidoa and Doloow as shown below. In this regards, the correlations need to be taken carefuly and could sometimes just come from thee fact that the categories are just coming from different regions.')
		elif database=='AVSI':
			continues=pickle.load( open( "cont_feat_avsi.p", "rb" ) )
			cat_cols=pickle.load( open( "cat_cols_avsi.p", "rb" ) )
				
		quests=correl[correl['variable_x'].fillna('').apply(lambda x: True if 'region' not in x else False)]
		
		#st.write(quest)
		#st.write(codes)
		#st.write(cat_cols)
		
		#st.write(data['assistancetype'].value_counts())
		
			
							
		for i in quests['variable_x'].unique():
			
			k=0
			st.markdown("""---""")		
			
			quest=quests[quests['variable_x']==i]
			
			#st.write(quest)
			
			if len(quest)>1 or 'bar' in quest['graphtype'].unique():
				col1,col2=st.columns([1,1])
			
			for i in range(len(quest)):
				
				
				
				#st.write(quest.iloc[i]['variable_x']+'##')
				#st.write('in cat cols: ',quest.iloc[i]['variable_x'] in cat_cols)
				
			
								
			
				if quest.iloc[i]['variable_x'] in cat_cols or quest.iloc[i]['variable_y'] in cat_cols:
					
					if quest.iloc[i]['variable_x'] in cat_cols:
						cat,autre=quest.iloc[i]['variable_x'],quest.iloc[i]['variable_y']
					else:
						cat,autre=quest.iloc[i]['variable_y'],quest.iloc[i]['variable_x']
					#st.write('cat: ',cat,' et autre: ',autre)
						
					df=pd.DataFrame(columns=[cat,autre])
					
					catcols=[j for j in data.columns if cat in j]
					cats=[' '.join(i.split(' ')[1:])[:57] for i in catcols]
				
					for n in range(len(catcols)):
						ds=data[[catcols[n],autre]].copy()
						ds=ds[ds[catcols[n]].isin(['Yes',1])]
						ds[catcols[n]]=ds[catcols[n]].apply(lambda x: cats[n])
						ds.columns=[cat,autre]
						df=df.append(ds)
					df['persons']=np.ones(len(df))		
					#st.write(df)		
					
					#st.write(quest.iloc[i]['graphtype'])
						
									
				else:	
					df=data[[quest.iloc[i]['variable_x'],quest.iloc[i]['variable_y']]].copy()
					df['persons']=np.ones(len(df))
				
				if quest.iloc[i]['graphtype']=='sunburst':
					st.subheader(quest.iloc[i]['title'])
					fig = px.sunburst(df.fillna(''), path=[quest.iloc[i]['variable_x'], quest.iloc[i]['variable_y']], 	values='persons',color=quest.iloc[i]['variable_y'])
					#fig.update_layout(title_text=quest.iloc[i]['variable_x'] + ' and ' +quest.iloc[i]['variable_y'],font=dict(size=20))
					st.plotly_chart(fig,size=1000)
					
					
					
				
				elif quest.iloc[i]['graphtype']=='treemap':
					
					st.subheader(quest.iloc[i]['title'])
					fig=px.treemap(df, path=[quest.iloc[i]['variable_x'], quest.iloc[i]['variable_y']], values='persons')
					#fig.update_layout(title_text=quest.iloc[i]['title'],font=dict(size=20))
					
					st.plotly_chart(fig,use_container_width=True)
					st.write(quest.iloc[i]['description'])
					
				
					
				elif quest.iloc[i]['graphtype']=='violin':
					
					#st.write(quest.iloc[i]['variable_x'],quest.iloc[i]['variable_y'])
					
					fig = go.Figure()
				
					if quest.iloc[i]['variable_x'].split(' ')[0] in codes['list name'].unique():
						categs = codes[codes['list name']==quest.iloc[i]['variable_x'].split(' ')[0]].sort_values(by='coding')['label'].tolist()				
					
					else:
						categs = df[quest.iloc[i]['variable_x']].unique()
					
					#st.write(categs)
					
					for categ in categs:
					    fig.add_trace(go.Violin(x=df[quest.iloc[i]['variable_x']][df[quest.iloc[i]['variable_x']] == categ],
		                         		y=df[quest.iloc[i]['variable_y']][df[quest.iloc[i]['variable_x']] == categ],
		                            		name=categ,
		                            		box_visible=True,
        	                   			meanline_visible=True,points="all",))
					
					
					fig.update_layout(showlegend=False)
					fig.update_yaxes(range=[-0.1, df[quest.iloc[i]['variable_y']].max()+1],title=quest.iloc[i]['ytitle'])
					k+=1
					
					
					
					if len(quest[quest['graphtype']=='violin'])>2:
						
						if k%2==1:
							col1.subheader(quest.iloc[i]['title'])
							col1.plotly_chart(fig,use_container_width=True)
							
						else:
							col2.subheader(quest.iloc[i]['title'])
							col2.plotly_chart(fig,use_container_width=True)
							
					else:
						st.subheader(quest.iloc[i]['title'])
						st.plotly_chart(fig,use_container_width=True)
					st.write(quest.iloc[i]['description'])
					
									
				elif quest.iloc[i]['graphtype']=='bar':
					
					#st.write(df[quest.iloc[i]['variable_y']].dtype)
					
					
					st.subheader(quest.iloc[i]['title'])
				
					col1,col2=st.columns([1,1])

					fig1=count2(quest.iloc[i]['variable_x'],quest.iloc[i]['variable_y'],\
					df,codes,legendtitle=quest.iloc[i]['legendtitle'],xaxis=quest.iloc[i]['xtitle'])
					
					col1.plotly_chart(fig1,use_container_width=True)
						
					fig2=pourcent2(quest.iloc[i]['variable_x'],quest.iloc[i]['variable_y'],\
					df,codes,legendtitle=quest.iloc[i]['legendtitle'],xaxis=quest.iloc[i]['xtitle'])
					#fig2.update_layout(title_text=quest.iloc[i]['title'],font=dict(size=20),showlegend=True,xaxis_tickangle=45)
					col2.plotly_chart(fig2,use_container_width=True)
					st.write(quest.iloc[i]['description'])
					#st.write(df)
						
	
			
	
		
	elif topic=='Display Sankey Graphs':
		
		if database=='COOPI':
		
			title1.title('Visuals for questions related to cultures (questions C3 to C17 and potentially others)')
			st.title('')
		
			
			crops=[i for i in data if i[0]=='B' and i[:3] not in ['B1_','B20','B19']]
			#st.write(sankey)	
		
			data_all=data[sankey].copy()
		
			#st.write()
		
			rest=[i for i in sankey if i not in crops]
			sankeyseeds=sankey[:65]
			sank=data[sankeyseeds]
		
			cowpea=data_all[rest+[i for i in crops if 'Cowpea' in i]].copy()
			sorghum=data_all[rest+[i for i in crops if 'Sorghum' in i]].copy()
			melon=data_all[rest+[i for i in crops if 'Melon' in i]].copy()
			maize=data_all[rest+[i for i in crops if 'Maize' in i]].copy()
			cabbage=data_all[rest+[i for i in crops if 'Cabbage' in i]].copy()
			tomatoes=data_all[rest+[i for i in crops if 'Tomatoes' in i]].copy()
			other=data_all[rest+[i for i in crops if 'other' in i]].copy()
		
			#st.write(cowpea)
		
			colonnes=['Seeds Planted','Type of seeds','Origin of seeds','Did you have adequate/enough seed',\
        	  'Origin of fertilizer']
			for i in [cowpea,sorghum,melon,maize,cabbage,tomatoes,other]:
    				i.columns=rest+colonnes
			cowpea=cowpea[cowpea['Seeds Planted']=='Yes']
			sorghum=sorghum[sorghum['Seeds Planted']=='Yes']
			melon=melon[melon['Seeds Planted']=='Yes']
			maize=maize[maize['Seeds Planted']=='Yes']
			cabbage=cabbage[cabbage['Seeds Planted']=='Yes']
			tomatoes=tomatoes[tomatoes['Seeds Planted']=='Yes']
			other=other[other['Seeds Planted']=='Yes']
		
			cowpea['Seeds Planted']=cowpea['Seeds Planted'].apply(lambda x: 'Cowpea')
			sorghum['Seeds Planted']=sorghum['Seeds Planted'].apply(lambda x: 'Sorghum')
			melon['Seeds Planted']=melon['Seeds Planted'].apply(lambda x: 'Melon')
			maize['Seeds Planted']=maize['Seeds Planted'].apply(lambda x: 'Maize')
			cabbage['Seeds Planted']=cabbage['Seeds Planted'].apply(lambda x: 'Cabbage')
			tomatoes['Seeds Planted']=tomatoes['Seeds Planted'].apply(lambda x: 'Tomatoes')
			other['Seeds Planted']=other['Seeds Planted'].apply(lambda x: 'Other')
		
			sank=pd.DataFrame(columns=rest+colonnes)
			for i in [cowpea,sorghum,melon,maize,cabbage,tomatoes,other]:
			    sank=sank.append(i)
			sank['ones']=np.ones(len(sank))
		
			st.title('Some examples')
		
			st.markdown("""---""")
			st.write('Origin of Seeds - Type of Seeds - '+questions['productivity_increased'])
			fig=sankey_graph(sank,['Origin of seeds','Type of seeds','Type of seeds'],height=600,width=1500)
			fig.update_layout(plot_bgcolor='black', paper_bgcolor='grey', width=1500)
			
			st.plotly_chart(fig,use_container_width=True)
			st.write('We can see that improved seeds and seeds received from AVSI improved the productivity')
		
			st.markdown("""---""")
			st.write(questions['B20_AVSI_equip']+' - '+questions['A15 Agricultural support']+' - '+questions['productivity_increased'])
			fig=sankey_graph(sank,['B20_AVSI_equip','A15 Agricultural support','productivity_increased'],height=600,width=1500)
			fig.update_layout(plot_bgcolor='black', paper_bgcolor='grey', width=1500)
			
			st.plotly_chart(fig,use_container_width=True)
			st.write('We can see that almost all those who received agricultural support and/or equipments improved the productivity')
			
		
			if st.checkbox('Design my own Sankey Graph'):
			
				st.markdown("""---""")
				feats=st.multiselect('Select features you want to see in the order you want them to appear', [questions[i] for i in rest]+colonnes)
			
				if len(feats)>=2:
					st.write(' - '.join(feats))
					a=False
					for i in feats:
						if i in colonnes:
							a=True
					if a:
						df=sank.copy()
					else:
						df=data_all
				
					features=[]
					for i in feats:
						if i in colonnes:
							features.append(i)
						else:
							features.append([n for n in questions if questions[n]==i][0])
				
					#st.write(features)
				
					fig3=sankey_graph(df,features,height=600,width=1500)
					fig3.update_layout(plot_bgcolor='black', paper_bgcolor='grey', width=1500)
					st.plotly_chart(fig3,use_container_width=True)
		
		if database=='AVSI':
			pass
		
		
		
	elif topic=='Display Wordclouds':
		
		if database=='COOPI':		
			continues=pickle.load( open( "cont_feat_coopi.p", "rb" ) )
			text=pickle.load( open( "text_coopi.p", "rb" ) )
		elif database=='AVSI':		
			continues=pickle.load( open( "cont_feat_avsi.p", "rb" ) )
			text=pickle.load( open( "text_avsi.p", "rb" ) )
		
		
		text2=[questions[i] for i in text]
		
		#st.write(questions)
		
		x, y = np.ogrid[100:500, :600]
		mask = ((x - 300)/2) ** 2 + ((y - 300)/3) ** 2 > 100 ** 2
		mask = 255 * mask.astype(int)
	
		courses,child=False,False
		title1.title('Wordclouds for open questions')
		
		feature=st.sidebar.selectbox('Select the question for which you would like to visualize wordclouds of answers',[i for i in text2])
		
		var=[i for i in questions if questions[i]==feature][0]
		
		col1, col2, col3 = st.columns([1,4,1])
		col2.title('Wordcloud from question:')
		col2.title(feature)
				
		
		corpus=' '.join(data[var].apply(lambda x:'' if x=='0' else x))
		corpus=re.sub('[^A-Za-z ]',' ', corpus)
		corpus=re.sub('\s+',' ', corpus)
		corpus=corpus.lower()
		
		col3.title('')
		col3.title('')
		col3.title('')
		sw=col3.multiselect('Select words you would like to remove from the wordcloud \n\n', [i[0] for i in Counter(corpus.split(' ')).most_common() if i[0] not in STOPWORDS][:20])
		
		if corpus==' ':
	    		corpus='No_response'
		else:
			corpus=' '.join([i for i in corpus.split(' ') if i not in sw])
		
		wc = WordCloud(background_color="#0E1117", repeat=False, mask=mask)		
		wc.generate(corpus)
		col2.image(wc.to_array(), use_column_width = True)	
		
		if col2.checkbox('Would you like to filter Wordcloud according to other questions'):
		
			feature2=col2.selectbox('Select one question to filter the wordcloud',[questions[i] for i in questions if i not in text])		
			filter2=[i for i in questions if questions[i]==feature2][0]
			
			if filter2 in continues:
				mini=int(data[filter2].fillna(0).min())
				maxi=int(data[filter2].fillna(0).max())
				minimum=col2.slider('Select the minimum value you want to visulize', min_value=mini,max_value=maxi)
				maximum=col2.slider('Select the maximum value you want to visulize', min_value=minimum,max_value=maxi+1)
				df=data[(data[filter2]>=minimum)&(data[filter2]<=maximum)].copy()	
				
			
			else:
				filter3=col2.multiselect('Select the responses you want to include', [i for i in data[filter2].unique()])
				df=data[data[filter2].isin(filter3)].copy()
			
			corpus=' '.join(df[var].apply(lambda x:'' if x=='0' else x))
			corpus=re.sub('[^A-Za-z ]',' ', corpus)
			corpus=re.sub('\s+',' ', corpus)
			corpus=corpus.lower()
			
			if corpus==' ' or corpus=='':
    				corpus='No_response'
			else:
				corpus=' '.join([i for i in corpus.split(' ') if i not in sw])
		
			wc = WordCloud(background_color="#0E1117", repeat=False, mask=mask)
			wc.generate(corpus)
			col2.image(wc.to_array(), use_column_width = True)
		
		
		
		
		
				
	
	
	

    
 
if __name__== '__main__':
    main()




    
