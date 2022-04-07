"""Application : Dashboard de Crédit Score

"""

# ====================================================================
# Chargement des librairies
# ====================================================================
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import pickle
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
import shap
from st_aggrid import AgGrid

# Répertoire du dashboard
# Test set brut original
FILE_APPLICATION_TEST = 'data/application_test.pickle'
# Test set pré-procédé
FILE_TEST_SET = 'data/test_set.pickle'
# Dashboard
FILE_DASHBOARD = 'data/df_dashboard.pickle'
# Client
FILE_CLIENT_INFO = 'data/df_info_client.pickle'
FILE_CLIENT_PRET = 'data/df_pret_client.pickle'

# Shap values
FILE_SHAP_VALUES = 'data/shap_values.pickle'

# Description
FILE_DESCRIPTION_VALUES = 'data/feature_description.pickle'

# ====================================================================
# IMAGES
# ====================================================================
# Logo de l'entreprise
logo =  Image.open("images/logo.png") 


# ====================================================================
# HEADER - TITRE
# ====================================================================
html_header="""
    <head>
        <title>Application Dashboard Credit Score</title>
       
    </head>             
    <h1 style="font-size:300%; color:Crimson; font-family:Arial"> Prêt à dépenser <br>
        <h2 style="color:Gray; font-family:Arial"> DASHBOARD</h2>
        
    </h1>
"""
st.set_page_config(page_title="Prêt à dépenser ", page_icon="", layout="wide")
st.markdown('<style>body{background-color: #fbfff0}</style>',unsafe_allow_html=True)
st.markdown(html_header, unsafe_allow_html=True)

# ====================================================================
# CHARGEMENT DES DONNEES
# ====================================================================

def load():
    with st.spinner('Uploading data'):
        
        # Import du dataframe des informations des traits stricts du client
        with open(FILE_CLIENT_INFO, 'rb') as df_info_client:
            df_info_client = pickle.load(df_info_client)
            
        # Import du dataframe des informations sur le prêt du client
        with open(FILE_CLIENT_PRET, 'rb') as df_pret_client:
            df_pret_client = pickle.load(df_pret_client)
            

            
       
        # Import du dataframe des informations sur le dashboard
        with open(FILE_DASHBOARD, 'rb') as df_dashboard:
            df_dashboard = pickle.load(df_dashboard)

       
        # Import du dataframe du test set nettoyé et pré-procédé
        with open(FILE_TEST_SET, 'rb') as df_test_set:
            test_set = pickle.load(df_test_set)

        # Import du dataframe du test set brut original
        with open(FILE_APPLICATION_TEST, 'rb') as df_application_test:
            application_test = pickle.load(df_application_test)

        
        # Import du dataframe du test set brut original
        with open(FILE_SHAP_VALUES, 'rb') as shap_values:
            shap_values = pickle.load(shap_values)

        # Import du dataframe du test set brut original
        with open(FILE_DESCRIPTION_VALUES, 'rb') as desc_values:
            desc_values = pickle.load(desc_values)
          
         
    return df_info_client, df_pret_client, df_dashboard,  test_set, \
            application_test, shap_values, desc_values

# Chargement des dataframes et du modèle
df_info_client, df_pret_client, df_dashboard,  test_set, \
            application_test, shap_values, desc_values = load()

# ====================================================================
# CHARGEMENT DE MODELE
# ====================================================================
best_model = pickle.load(open('model.pickle', 'rb'))


# ====================================================================
# CHOIX DU CLIENT
# ====================================================================

html_select_client="""
    <div class="card">
      <div class="card-body" style="border-radius: 10px 10px 0px 0px;
                  background: #DEC7CB; padding-top: 5px; width: auto;
                  height: 40px;">
        <h3 class="card-title" style="background-color:#DEC7CB; color:green;
                   font-family:Georgia; text-align: center; padding: 0px 0;">
          Client information
        </h3>
      </div>
    </div>
    """

st.markdown(html_select_client, unsafe_allow_html=True)

with st.container():

    col1, col2 = st.columns([1,4])
    with col1:
        st.write("")
        col1.header("**Client ID**")
        client_id = col1.selectbox('Insert client ID :',
                                   test_set['SK_ID_CURR'].unique())
    with col2:
        # Infos principales client
        # st.write("*Traits stricts*")
        client_info = df_info_client[df_info_client['SK_ID_CURR'] == client_id].iloc[: , :]
        client_info.set_index('SK_ID_CURR', inplace=True)
        st.table(client_info)
        # Infos principales sur la demande de prêt
        # st.write("*Demande de prêt*")
        client_pret = df_pret_client[df_pret_client['SK_ID_CURR'] == client_id].iloc[:, :]
        client_pret.set_index('SK_ID_CURR', inplace=True)
        st.table(client_pret)



# ============== Score du client en pourcentage ==> en utilisant le modèle ======================
# Sélection des variables du clients
X_test = test_set[test_set['SK_ID_CURR'] == client_id]
# Score des prédictions de probabiltés
y_proba = best_model.predict_proba(X_test.drop('SK_ID_CURR', axis=1))[:, 1]
# Score du client en pourcentage arrondi et nombre entier
score_client = int(np.rint(y_proba * 100))



# Graphique de jauge du cédit score ==========================================
fig_jauge = go.Figure(go.Indicator(
    mode = 'gauge+number',
    # Score du client en % df_dashboard['SCORE_CLIENT_%']
    value = score_client,  
    domain = {'x': [0, 1], 'y': [0, 1]},
    title = {'text': 'Client\'s Credit Score', 'font': {'size': 24}},
    # Score des 10 voisins test set
    # df_dashboard['SCORE_10_VOISINS_MEAN_TEST']
    delta = {
             'increasing': {'color': 'Crimson'},
             'decreasing': {'color': 'Green'}},
    gauge = {'axis': {'range': [None, 100],
                      'tickwidth': 3,
                      'tickcolor': 'darkblue'},
             'bar': {'color': 'white', 'thickness' : 0.25},
             'bgcolor': 'white',
             'borderwidth': 2,
             'bordercolor': 'gray',
             'steps': [{'range': [0, 25], 'color': 'Green'},
                       {'range': [25, 49.49], 'color': 'LimeGreen'},
                       {'range': [49.5, 50.5], 'color': 'red'},
                       {'range': [50.51, 75], 'color': 'Orange'},
                       {'range': [75, 100], 'color': 'Crimson'}],
             'threshold': {'line': {'color': 'white', 'width': 10},
                           'thickness': 0.8,
                           # Score du client en %
                           # df_dashboard['SCORE_CLIENT_%']
                           'value': score_client}}))

fig_jauge.update_layout(paper_bgcolor='white',
                        height=400, width=500,
                        font={'color': 'darkblue', 'family': 'Arial'},
                        margin=dict(l=0, r=0, b=0, t=0, pad=0))

with st.container():
    # JAUGE + récapitulatif du score moyen des voisins
    col1, col2 = st.columns([1.5, 1])
    with col1:
        st.plotly_chart(fig_jauge)
    with col2:
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        # Texte d'accompagnement de la jauge
        if 0 <= score_client < 25:
            score_text = 'Credit score : EXCELLENT'
            st.success(score_text)
        elif 25 <= score_client < 50:
            score_text = 'Credit score : BON'
            st.success(score_text)
        elif 50 <= score_client < 75:
            score_text = 'Credit score : MOYEN'
            st.warning(score_text)
        else :
            score_text = 'Credit score : BAS'
            st.error(score_text)
        st.write("")    
        #st.markdown(f'Crédit score moyen des 10 clients similaires : **{score_moy_voisins_test}**')
        #st.markdown(f'**{pourc_def_voisins_train}**% de clients voisins réellement défaillants dans l\'historique')
        #st.markdown(f'**{pourc_def_voisins_test}**% de clients voisins défaillants prédits pour les nouveaux clients')
  

# LOGO
# --------------------------------------------------------------------
# Chargement du logo de l'entreprise
st.sidebar.image(logo, width=240,  use_column_width='always')


# ====================================================================
# SIDEBAR
# ====================================================================
# Toutes Les informations non modifiées du client courant
df_client_origin = application_test[application_test['SK_ID_CURR'] == client_id]

# Toutes Les informations non modifiées du client courant
df_client_test = test_set[test_set['SK_ID_CURR'] == client_id]

# Les informations pré-procédées du client courant
df_client_courant = df_dashboard[df_dashboard['SK_ID_CURR'] == client_id]


# --------------------------------------------------------------------
# MORE INFORMATIONS
# --------------------------------------------------------------------
def all_infos_clients():
    ''' Affiche toutes les informations sur le client courant
    '''
    html_all_infos_clients="""
        <div class="card">
            <div class="card-body" style="border-radius: 10px 10px 0px 0px;
                  background: #DEC7CB; padding-top: 5px; width: auto;
                  height: 40px;">
                  <h3 class="card-title" style="background-color:#DEC7CB; color:green;
                      font-family:Georgia; text-align: center; padding: 0px 0;">
                      All client's Information
                  </h3>
            </div>
        </div>
        """
    
    # ====================== GRAPHIQUES COMPARANT CLIENT COURANT / CLIENTS SIMILAIRES =========================== 
    if st.sidebar.checkbox("Client"):     
        
        st.markdown(html_all_infos_clients, unsafe_allow_html=True)

        with st.spinner('**Waitting...**'):                    
            with st.expander('All informations of the client', expanded=True):                        
                AgGrid(df_client_test, height=100, fit_columns_on_grid_load=False)
                #st.table( df_client_test)
                #st.dataframe( df_client_origin)
                AgGrid(df_client_origin, height=100, fit_columns_on_grid_load=False)


            with st.expander('Description of all information',expanded=True):
                st.dataframe(desc_values)


st.sidebar.subheader('More information :')
all_infos_clients()
                
    

# ====================================================================
# HTML MARKDOWN
# ====================================================================
html_YEAR_BIRTH = "<h4 style='text-align: center'>Age (year) </h4><hr/>" 
html_AMT_ANNUITY = "<h4 style='text-align: center'>AMT of ANNUITY</h4> <hr/>"
html_YEAR_EMPLOYED = "<h4 style='text-align: center'>Experience (yrear)</h4> <hr/>" 
html_OWN_CAR_AGE = "<h4 style='text-align: center'>Age of own car</h4> <hr/>" 
html_AMT_CREDIT = "<h4 style='text-align: center'>AMT of CREDIT</h4><hr/>" 
html_OWN_CAR_AGE = "<h4 style='text-align: center'>Car'/s age of client</h4><hr/>" 
html_EXT_SOURCE_Mean = "<h4 style='text-align: center'>Mean of 3 Externel sources</h4><hr/>" 

html_EXT_SOURCE_1 = "<h4 style='text-align: center'>EXT_SOURCE_3</h4> <br/> <h5 style='text-align: center'>Source externe normalisée (valeur * 100)</h5> <hr/>" 


# --------------------------------------------------------------------
# SATATISTICS
# --------------------------------------------------------------------
def stat_graph():
    ''' Affiche les les graphiques du client courant
    '''
    html_stat_graph="""
        <div class="card">
            <div class="card-body" style="border-radius: 10px 10px 0px 0px;
                  background: #DEC7CB; padding-top: 5px; width: auto;
                  height: 40px;">
                  <h3 class="card-title" style="background-color:#DEC7CB; color:green;
                      font-family:Georgia; text-align: center; padding: 0px 0;">
                      Distribution
                  </h3>
            </div>
        </div>
        """
    titre = True

    
         
    # ====================== GRAPHIQUES DE SITUATION CLIENT COURANT  =========================== 
    if st.sidebar.checkbox("statistics"):


        st.markdown(html_stat_graph, unsafe_allow_html=True)
                
        with st.spinner('**Waitting...**'):                 
                       
            with st.expander('Age of clients',expanded=False):


                with st.container():
                                        
                    df_dashboard['YEAR_BIRTH'] = \
                        np.trunc(np.abs(df_dashboard['DAYS_BIRTH'] / 365)).astype('int8')
                    age_client = int(df_dashboard[df_dashboard['SK_ID_CURR'] == client_id]['YEAR_BIRTH'].values)         
                    st.markdown(html_YEAR_BIRTH, unsafe_allow_html=True)

                    # ==================== ViolinPlot ========================================================
                    st.set_option('deprecation.showPyplotGlobalUse', False)
                    sns.violinplot(x='PRED_CLASSE_CLIENT', y='YEAR_BIRTH',
                                   data=df_dashboard,palette=['LimeGreen', 'Crimson'])

                    plt.plot(df_client_courant['PRED_CLASSE_CLIENT'],age_client,
                             color="orange", marker="$\\bigotimes$", markersize=18)

                    plt.xlabel('TARGET', fontsize=10)
                    client = mlines.Line2D([], [], color='orange', marker='$\\bigotimes$',linestyle='None',
                                               markersize=10, label='Clent\'s position')

                    plt.legend(handles=[client], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                    st.pyplot(showPyplotGlobalUse = False)
                    st.set_option('deprecation.showPyplotGlobalUse', False)    

                    # ==================== DistPlot ==========================================================
                    # No_Defaulter
                    sns.distplot(df_dashboard['YEAR_BIRTH'][df_dashboard['PRED_CLASSE_CLIENT'] == 0],
                                 label='No-Defaulter', hist=False, color='Green')
                    # Défaillants
                    sns.distplot(df_dashboard['YEAR_BIRTH'][df_dashboard['PRED_CLASSE_CLIENT'] == 1],
                                 label='Defaulter', hist=False, color='Crimson')
                    plt.xlabel('YEAR_BIRTH', fontsize=16)
                    plt.ylabel('Probability Density', fontsize=16)
                    plt.xticks(fontsize=16, rotation=90)
                    plt.yticks(fontsize=16)
                    # Position du client
                    plt.axvline(x=age_client, color='orange', label='Clent\'s position')
                    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=18)
                    st.pyplot()                                   
                                                                             
 


                # ==============================================================
                # Variable Annuity ($)
                # ==============================================================
            with st.expander('Annuity ($)', expanded=False):
                            
                with st.spinner('**Waitting**'):

                    ann_client = int(df_dashboard[df_dashboard['SK_ID_CURR'] == client_id]['AMT_ANNUITY'].values)         
                    st.markdown(html_AMT_ANNUITY, unsafe_allow_html=True)

                    # ==================== ViolinPlot ========================================================
                    st.set_option('deprecation.showPyplotGlobalUse', False)
                    sns.violinplot(x='PRED_CLASSE_CLIENT', y='AMT_ANNUITY',
                                   data=df_dashboard,palette=['LimeGreen', 'Crimson'])

                    plt.plot(df_client_courant['PRED_CLASSE_CLIENT'],ann_client,
                             color="orange", marker="$\\bigotimes$", markersize=18)

                    plt.xlabel('TARGET', fontsize=10)
                    client = mlines.Line2D([], [], color='orange', marker='$\\bigotimes$',linestyle='None',
                                               markersize=10, label='Clent\'s position')

                    plt.legend(handles=[client], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                    st.pyplot(showPyplotGlobalUse = False)
                    st.set_option('deprecation.showPyplotGlobalUse', False)    

                    # ==================== DistPlot ==========================================================
                    # No-Defaulter
                    sns.distplot(df_dashboard['AMT_ANNUITY'][df_dashboard['PRED_CLASSE_CLIENT'] == 0],
                                 label='No-Defaulter', hist=False, color='Green')
                    # Défaillants
                    sns.distplot(df_dashboard['AMT_ANNUITY'][df_dashboard['PRED_CLASSE_CLIENT'] == 1],
                                 label='Defaulter', hist=False, color='Crimson')
                    plt.xlabel('AMT_ANNUITY', fontsize=16)
                    plt.ylabel('Probability Density', fontsize=16)
                    plt.xticks(fontsize=16, rotation=90)
                    plt.yticks(fontsize=16)
                    # Position du client
                    plt.axvline(x=ann_client, color='orange', label='Clent\'s position')
                    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=18)
                    st.pyplot()  
            

                # ==============================================================
                # Variable YEAR_EMPLOYED
                # ==============================================================
            with st.expander('Experience ', expanded=False):
                            
                with st.spinner('**Waitting**'):

                    ann_client = int(df_dashboard[df_dashboard['SK_ID_CURR'] == client_id]['YEAR_EMPLOYED'].values)         
                    st.markdown(html_YEAR_EMPLOYED, unsafe_allow_html=True)

                    # ==================== ViolinPlot ========================================================
                    st.set_option('deprecation.showPyplotGlobalUse', False)
                    sns.violinplot(x='PRED_CLASSE_CLIENT', y='YEAR_EMPLOYED',
                                   data=df_dashboard,palette=['LimeGreen', 'Crimson'])

                    plt.plot(df_client_courant['PRED_CLASSE_CLIENT'],ann_client,
                             color="orange", marker="$\\bigotimes$", markersize=18)

                    plt.xlabel('TARGET', fontsize=10)
                    client = mlines.Line2D([], [], color='orange', marker='$\\bigotimes$',linestyle='None',
                                               markersize=10, label='Clent\'s position')

                    plt.legend(handles=[client], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                    st.pyplot(showPyplotGlobalUse = False)
                    st.set_option('deprecation.showPyplotGlobalUse', False)    

                    # ==================== DistPlot ==========================================================
                    # No-Defaulter
                    sns.distplot(df_dashboard['YEAR_EMPLOYED'][df_dashboard['PRED_CLASSE_CLIENT'] == 0],
                                 label='No-Defaulter', hist=False, color='Green')
                    # Défaillants
                    sns.distplot(df_dashboard['YEAR_EMPLOYED'][df_dashboard['PRED_CLASSE_CLIENT'] == 1],
                                 label='Defaulter', hist=False, color='Crimson')
                    plt.xlabel('YEAR_EMPLOYED', fontsize=16)
                    plt.ylabel('Probability Density', fontsize=16)
                    plt.xticks(fontsize=16, rotation=90)
                    plt.yticks(fontsize=16)
                    # Position du client
                    plt.axvline(x=ann_client, color='orange', label='Clent\'s position')
                    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=18)
                    st.pyplot()




                    # ==============================================================
                # Variable AMT_CREDIT
                # ==============================================================
            with st.expander('Amounth of credit', expanded=False):
                            
                with st.spinner('**Waitting**'):

                    crd_client = int(df_dashboard[df_dashboard['SK_ID_CURR'] == client_id]['AMT_CREDIT'].values)         
                    st.markdown(html_AMT_CREDIT, unsafe_allow_html=True)

                    # ==================== ViolinPlot ========================================================
                    st.set_option('deprecation.showPyplotGlobalUse', False)
                    sns.violinplot(x='PRED_CLASSE_CLIENT', y='AMT_CREDIT',
                                   data=df_dashboard,palette=['LimeGreen', 'Crimson'])

                    plt.plot(df_client_courant['PRED_CLASSE_CLIENT'],crd_client,
                             color="orange", marker="$\\bigotimes$", markersize=18)

                    plt.xlabel('TARGET', fontsize=10)
                    client = mlines.Line2D([], [], color='orange', marker='$\\bigotimes$',linestyle='None',
                                               markersize=10, label='Clent\'s position')

                    plt.legend(handles=[client], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                    st.pyplot(showPyplotGlobalUse = False)
                    st.set_option('deprecation.showPyplotGlobalUse', False)    

                    # ==================== DistPlot ==========================================================
                    # No-Defaulter
                    sns.distplot(df_dashboard['AMT_CREDIT'][df_dashboard['PRED_CLASSE_CLIENT'] == 0],
                                 label='No-Defaulter', hist=False, color='Green')
                    # Défaillants
                    sns.distplot(df_dashboard['AMT_CREDIT'][df_dashboard['PRED_CLASSE_CLIENT'] == 1],
                                 label='Defaulter', hist=False, color='Crimson')
                    plt.xlabel('AMT_CREDIT', fontsize=16)
                    plt.ylabel('Probability Density', fontsize=16)
                    plt.xticks(fontsize=16, rotation=90)
                    plt.yticks(fontsize=16)
                    # Position du client
                    plt.axvline(x=crd_client, color='orange', label='Clent\'s position')
                    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=18)
                    st.pyplot()  
            

                # ==============================================================
                # Variable EXT_SOURCE_3
                # ==============================================================
            with st.expander('OWN_CAR_AGE', expanded=False):
                            
                with st.spinner('**Waitting**'):

                    car_age = float(df_dashboard[df_dashboard['SK_ID_CURR'] == client_id]['OWN_CAR_AGE'].values)         
                    st.markdown(html_OWN_CAR_AGE, unsafe_allow_html=True)

                    # ==================== ViolinPlot ========================================================
                    st.set_option('deprecation.showPyplotGlobalUse', False)
                    sns.violinplot(x='PRED_CLASSE_CLIENT', y='OWN_CAR_AGE',
                                   data=df_dashboard,palette=['LimeGreen', 'Crimson'])

                    plt.plot(df_client_courant['PRED_CLASSE_CLIENT'],car_age,
                             color="orange", marker="$\\bigotimes$", markersize=18)

                    plt.xlabel('TARGET', fontsize=10)
                    client = mlines.Line2D([], [], color='orange', marker='$\\bigotimes$',linestyle='None',
                                               markersize=10, label='Clent\'s position')

                    plt.legend(handles=[client], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                    st.pyplot(showPyplotGlobalUse = False)
                    st.set_option('deprecation.showPyplotGlobalUse', False)    

                    # ==================== DistPlot ==========================================================
                    # No-Defaulter
                    sns.distplot(df_dashboard['OWN_CAR_AGE'][df_dashboard['PRED_CLASSE_CLIENT'] == 0],
                                 label='No-Defaulter', hist=False, color='Green')
                    # Défaillants
                    sns.distplot(df_dashboard['OWN_CAR_AGE'][df_dashboard['PRED_CLASSE_CLIENT'] == 1],
                                 label='Defaulter', hist=False, color='Crimson')
                    plt.xlabel('OWN_CAR_AGE', fontsize=16)
                    plt.ylabel('Probability Density', fontsize=16)
                    plt.xticks(fontsize=16, rotation=90)
                    plt.yticks(fontsize=16)
                    # Position du client
                    plt.axvline(x=car_age, color='orange', label='Clent\'s position')
                    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=18)
                    st.pyplot() 


                # ==============================================================
                # Variable EXT_SOURCE_mean
                # ==============================================================
            with st.expander('EXT_sources_MEAN', expanded=False):
                            
                with st.spinner('**Waitting**'):

                    ext_m = float(df_dashboard[df_dashboard['SK_ID_CURR'] == client_id]['NEW_EXT_MEAN'].values)         
                    st.markdown(html_EXT_SOURCE_Mean, unsafe_allow_html=True)

                    # ==================== ViolinPlot ========================================================
                    st.set_option('deprecation.showPyplotGlobalUse', False)
                    sns.violinplot(x='PRED_CLASSE_CLIENT', y='NEW_EXT_MEAN',
                                   data=df_dashboard,palette=['LimeGreen', 'Crimson'])

                    plt.plot(df_client_courant['PRED_CLASSE_CLIENT'],ext_m,
                             color="orange", marker="$\\bigotimes$", markersize=18)

                    plt.xlabel('TARGET', fontsize=10)
                    client = mlines.Line2D([], [], color='orange', marker='$\\bigotimes$',linestyle='None',
                                               markersize=10, label='Clent\'s position')

                    plt.legend(handles=[client], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                    st.pyplot(showPyplotGlobalUse = False)
                    st.set_option('deprecation.showPyplotGlobalUse', False)    

                    # ==================== DistPlot ==========================================================
                    # No-Defaulter
                    sns.distplot(df_dashboard['NEW_EXT_MEAN'][df_dashboard['PRED_CLASSE_CLIENT'] == 0],
                                 label='No-Defaulter', hist=False, color='Green')
                    # Défaillants
                    sns.distplot(df_dashboard['NEW_EXT_MEAN'][df_dashboard['PRED_CLASSE_CLIENT'] == 1],
                                 label='Defaulter', hist=False, color='Crimson')
                    plt.xlabel('NEW_EXT_MEAN', fontsize=16)
                    plt.ylabel('Probability Density', fontsize=16)
                    plt.xticks(fontsize=16, rotation=90)
                    plt.yticks(fontsize=16)
                    # Position du client
                    plt.axvline(x=ext_m, color='orange', label='Clent\'s position')
                    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=18)
                    st.pyplot()



stat_graph()
          



# --------------------------------------------------------------------
# FACTEURS D'INFLUENCE : SHAP VALUE
# --------------------------------------------------------------------
    
def affiche_facteurs_influence():
    ''' Affiche les facteurs d'influence du client courant
    '''
    html_facteurs_influence="""
        <div class="card">
            <div class="card-body" style="border-radius: 10px 10px 0px 0px;
                  background: #DEC7CB; padding-top: 5px; width: auto;
                  height: 40px;">
                  <h3 class="card-title" style="background-color:#DEC7CB; color:green;
                      font-family:Georgia; text-align: center; padding: 0px 0;">
                      Importante variables
                  </h3>
            </div>
        </div>
        """
    
    # ====================== GRAPHIQUES Influential factors======================================== 
    if st.sidebar.checkbox("Influential factors"):     
        
        st.markdown(html_facteurs_influence, unsafe_allow_html=True)

        with st.spinner('**Waitting...**'):                 
            
                
                explainer = shap.TreeExplainer(best_model)
                
                client_index = test_set[test_set['SK_ID_CURR'] == client_id].index.item()
                X_shap = test_set.set_index('SK_ID_CURR')
                X_test_courant = X_shap.iloc[client_index]
                X_test_courant_array = X_test_courant.values.reshape(1, -1)
                
                shap_values_courant = explainer.shap_values(X_test_courant_array)
                
                col1, col2 = st.columns([1, 1])
                # BarPlot du client courant
                with col1:

                    plt.clf()
                    

                    # BarPlot du client courant
                    shap.plots.bar( shap_values[client_index], max_display=40)
                    
                    fig = plt.gcf()
                    fig.set_size_inches((10, 20))
                    # Plot the graph on the dashboard
                    st.pyplot(fig)
     
                # Decision plot du client courant
                with col2:
                    plt.clf()

                    # Decision Plot
                    shap.decision_plot(explainer.expected_value[1], shap_values_courant[1],
                                    X_test_courant)
                
                    fig2 = plt.gcf()
                    fig2.set_size_inches((10, 15))
                    # Plot the graph on the dashboard
                    st.pyplot(fig2)
                    

affiche_facteurs_influence()





