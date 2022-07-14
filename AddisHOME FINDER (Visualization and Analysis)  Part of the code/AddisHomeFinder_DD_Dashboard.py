
# Dash Dashboard
# Functionalites include -- Based on a set of criteria, this project analyzes existing real estate competition analyses.
# Such as price, location etc ....and deliver interactive analysis
# Geletaw Sahle

import pandas as pd
import streamlit as st
import plotly.express as px
from PIL import Image
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

DATA_URL = (
    '/Users/geletawsahle/Desktop/CP_2020_WEPAPP/AmanMiko.csv')


@st.cache
# Function to call historical records
def load_data(nrows):
    # Function to call historical records

    # raw CP data
    data = pd.read_csv(DATA_URL, nrows=nrows)
    return data


def getImage(path, zoom=0.1):
    return OffsetImage(plt.imread(path), zoom=zoom)


image = Image.open('/Users/geletawsahle/Desktop/CP_2020_WEPAPP/AddisHomeLogo.png')
# st.sidebar.image(image, caption='', width=None)

st.subheader("eFinder: AddisHome Data Analysis and Visualization Tool")
data = load_data(1000)

# Visualize CP class from historical records
# st.write(data)

st.sidebar.subheader("AddisHome: eFinder")

activitesChoice = ["Preview",
                   "Visualization",
                   "Analysis",
                   "Competitor Analysis",
                   "Comparision",
                   "Next Buyer"]

activites = st.sidebar.selectbox("What do you want to uncover?", activitesChoice)

if activites == "Preview":
    st.info("Preview Records")
    filteringList = st.multiselect("Filter with", data.columns)
    st.write("No. of Selected Filtering Criteria:", len(filteringList))
    st.write("Details:", filteringList)

    # Values for flitering
    filteringListValues = {}
    for i in range(0, len(filteringList)):
        MsValue = st.selectbox(filteringList[i], data[filteringList[i]].unique())
        filteringListValues[filteringList[i]] = MsValue
    st.write("Flitering criteria with thier values:", filteringListValues)

    # Create your filtering function:

    def filter_dict(data, dic):
        return data.loc[data[list(filteringListValues.keys())].isin(list(filteringListValues.values())).all(axis=1), :]

    # Use it on your DataFrame:

    st.write("Loading data ...")
    st.write(filter_dict(data, filteringListValues))

    # pre-processing
    preprocessingType = ["Missing Value", "Replace", "Outlier", "NoisyValue"]
    preprocessing = st.selectbox("Preprocessing", preprocessingType)
    if preprocessing == "Missing Value":
        if data.isnull().values.any():
            data = data.fillna("Not Available ")
            st.write(data)
        else:
            st.write("There is no missing value on the given dataset")


elif activites == "Analysis":
    crossTabulationValue = ["Stacked Barchart", "Scatterplot", "Count"]
    st.write(
        '<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

    choice = st.radio("Choice:", crossTabulationValue)
    st.info("Cross Tabulation Analysis")
    if choice == "Stacked Barchart":
        # cols = ["Disease_Name", "CP"]
        # st_ms = st.multiselect("Columns", data.columns.tolist(), default=cols)
        # st.write(st_ms)

        # xx = data.groupby(['Disease_Name']).CP.value_counts(normalize=True).mul(100)
        #  st.write(xx)
        groupby = st.selectbox("Group by:", data.columns.tolist(), index=27)
        groupwith = st.multiselect("Group with", data.columns.tolist(),
                                   default=["የተሸጠው_ቤት_ስፋት", "የትዳር_ሁኔታ"])
        st.write(groupwith)
        tmp = data.groupby(groupby)[groupwith].count()
        st.info("Result: Count")
        st.dataframe(tmp)  #
        st.info("Visualization: Stacked Barchart")
        st.bar_chart(tmp)
    elif choice == "Scatterplot":
        data = data.fillna("Not Available")
        x = st.selectbox("X:", data.columns.tolist(), index=18)
        y = st.selectbox("Y:", data.columns.tolist(), index=19)
        color = st.selectbox("Visualize:", data.columns.tolist(), index=27)
        fig = px.scatter(data, x=x, y=y, color=color, title="")
        fig.update_traces(marker=dict(size=12,
                                      line=dict(width=2,
                                                color='DarkSlateGrey')),
                          selector=dict(mode='markers'))
        st.plotly_chart(fig)
    else:
        selectColumn = st.selectbox("Columns", data.columns.tolist(), index=18)
        aggregatedCount = pd.DataFrame(data[selectColumn].value_counts().reset_index().values,
                                       columns=["columnValue", "Aggregate"])
        st.write(aggregatedCount)

elif activites == "Visualization":
    st.info("Visualization Dashboard")
    select = st.selectbox('Vizualization type', ['Histogram', 'Pie Chart'], key='1', index=1)
    requiredVIZ = st.selectbox("Want to visualize:", data.columns, index=27)

    st.write("Visualization of ", requiredVIZ, "using", select)
    viz_count = data[requiredVIZ].value_counts()
    viz_count = pd.DataFrame({requiredVIZ: viz_count.index,
                              'Values': viz_count.values})

    if select == "Histogram":
        fig = px.bar(viz_count, x=requiredVIZ, y='Values', color='Values', height=500)
        st.plotly_chart(fig)
    else:
        fig = px.pie(viz_count, values='Values', names=requiredVIZ)
        st.plotly_chart(fig)

elif activites == "Competitor Analysis":
    AnalysisDATA_URL = (
        '/Users/geletawsahle/Desktop/CP_2020_WEPAPP/CompAnalysis.csv')
    data = pd.read_csv(AnalysisDATA_URL)

    # Allow use to choose
    columns = ['Price', 'Down_Per_Grade', 'LocationGrade', 'QualityGrade',
               'InterestGrade', 'FacilityGrade', 'Payment_Term', 'Delivery_Period']
    x_axis = st.selectbox('x_axis', columns)
    # data.columns.tolist(), index=3
    y_axis = st.selectbox('y_axis', columns)
    hover_name = st.selectbox('hover name', data.columns)

    if y_axis == 'InterestGrade' or x_axis == 'InterestGrade':
        data = data.dropna(subset=['InterestGrade'], how='any')
    # Marker = data.Real_Estate.unique()
    # st.write(Marker)

    fig = px.scatter(data,
                     x=x_axis,
                     y=y_axis,
                     hover_name=hover_name,
                     # text="Real Estate",
                     color=hover_name,

                     # animation_frame="Real_Estate",
                     title=f'{y_axis} vs. {x_axis}'
                     )

    symbols = {'Ayat': 'star-triangle-up',
               'Noah': 'star-triangle-up',
               'Gift': 'star-triangle-up',
               'Metropolitan': 'diamond',
               'Lagare': 'diamond',
               'Flintstone': 'pentagon',
               'Golden Art': 'square',
               'Tsehay': 'star-triangle-up',
               'Pluto': 'star-triangle-up',
               'Sunrise': 'diamond',
               'ZenebeFrew': 'arrow-up',
               'Enyi': 'star-triangle-up',
               'Get-As': 'pentagon-dot',
               'Mattes': 'star',
               'Evergrand': 'cross',
               'Jambo': 'cross',
               'Elilta': 'diamond',
               'AL-SAM': 'star',
               'Bright': 'pentagon',
               'J.H. SIMEX': 'diamond-wide-dot',
               'Mezaber': 'star-triangle-up',
               'Roha': 'star',
               'The Developer Group': 'square',
               'Saccure': 'diamond',
               'Champion Properties': 'star',
               'ETCOF': 'diamond-wide-dot',
               'FH': 'circle',
               'Kefita': 'diamond',
               }

    fig.add_vline(x=data[x_axis].astype(int).mean(), line_width=2,
                  line_color="red")
    fig.add_hline(y=data[y_axis].astype(int).mean(), line_width=2,
                  line_color="red")
    #  names
    fig.update_traces(textposition='top center',
                      marker=dict(size=20)  # , symbol='star-triangle-up'),
                      )
    #fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 2000
    # set all symbols in fig
    for i, d in enumerate(fig.data):
        fig.data[i].marker.symbol = symbols[fig.data[i].name]

    fig.update_layout(
        height=600,
        # transition={'duration': 6000}
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.3,
            xanchor="right",
            x=1
        )

        # title_text='GDP and Life Expectancy (Americas, 2007)'
    )
    st.plotly_chart(fig)

    if y_axis == 'LocationGrade' or x_axis == 'LocationGrade':
        location = [5, 4, 3, 2, 1]
        st.write(
            '<style>div.row-widget.stRadio > div{flex-direction:row;}</style>',
            unsafe_allow_html=True)
        choice = st.radio("Location Grade:", location)
        data1 = data[data['LocationGrade'] == choice]
        # st.write("Selected Dataset", data1)
        subfig = px.scatter(data1,
                            x=x_axis,
                            y=data1.QualityGrade,
                            hover_name=hover_name,
                            # text=data1.Real_Estate,
                            color=hover_name,
                            title=f'{y_axis} = {choice} vs. {x_axis}'
                            )

        # set all symbols in fig
        for i, d in enumerate(subfig.data):
            subfig.data[i].marker.symbol = symbols[subfig.data[i].name]

        subfig.add_vline(x=data1[x_axis].astype(int).mean(), line_width=2,
                         line_color="red")
        subfig.add_hline(y=data1['QualityGrade'].astype(int).mean(), line_width=2,
                         line_color="red")
        subfig.update_traces(textposition='top right',
                             marker=dict(size=20)  # ,  symbol='star-triangle-up'),
                             # marker={'size': 20}

                             )
        subfig.update_layout(
            # showlegend=False,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.4,
                xanchor="right",
                x=1
            )
        )
        st.plotly_chart(subfig)

        # display the logos
        paths = [
            '/Users/geletawsahle/Desktop/CP_2020_WEPAPP/Ayat.jpg',
            '/Users/geletawsahle/Desktop/CP_2020_WEPAPP/FH.jpg',
            '/Users/geletawsahle/Desktop/CP_2020_WEPAPP/Bright.jpg',
            '/Users/geletawsahle/Desktop/CP_2020_WEPAPP/Champion Properties.jpg',
            '/Users/geletawsahle/Desktop/CP_2020_WEPAPP/Enyi.jpg',
            '/Users/geletawsahle/Desktop/CP_2020_WEPAPP/Elilta.jpg',
            '/Users/geletawsahle/Desktop/CP_2020_WEPAPP/ETCOF.jpg',
            '/Users/geletawsahle/Desktop/CP_2020_WEPAPP/Evergrand.jpg',
            '/Users/geletawsahle/Desktop/CP_2020_WEPAPP/Flintstone.jpg',
            '/Users/geletawsahle/Desktop/CP_2020_WEPAPP/Getas.jpg',
            '/Users/geletawsahle/Desktop/CP_2020_WEPAPP/Metropolitan.jpg',
            '/Users/geletawsahle/Desktop/CP_2020_WEPAPP/Gift.jpg',
            '/Users/geletawsahle/Desktop/CP_2020_WEPAPP/Golden Art.jpg',
            '/Users/geletawsahle/Desktop/CP_2020_WEPAPP/JHSIMEX.jpg',
            '/Users/geletawsahle/Desktop/CP_2020_WEPAPP/Kefita.jpg',
            '/Users/geletawsahle/Desktop/CP_2020_WEPAPP/Lagare.jpg',

            # Q3
            '/Users/geletawsahle/Desktop/CP_2020_WEPAPP/Mattes.jpg',
            '/Users/geletawsahle/Desktop/CP_2020_WEPAPP/Mezaber.jpg',
            '/Users/geletawsahle/Desktop/CP_2020_WEPAPP/Noah.jpg',
            '/Users/geletawsahle/Desktop/CP_2020_WEPAPP/Saccure.jpg',
            '/Users/geletawsahle/Desktop/CP_2020_WEPAPP/Sunrise.jpg',
        ]
        # ploting using seaborn image symbols
        fig, ax = plt.subplots()
        ax = sns.scatterplot(data=data, x=x_axis, y=y_axis)

        # set all symbols on the figure
        # for i, d in enumerate(fig.data):
        #    fig.data[i].marker.symbol = paths[fig.data[i].name]
        plt.show()
        st.pyplot(fig)

        # Testing the logo based scatter plot

        fig, ax = plt.subplots()
        # Scatterplot
        ax = sns.scatterplot(data=data, x='Price', y='LocationGrade')

        # Title
        plt.title(f"Real State Comparision")

        # real state names
        # for i in range(data.shape[0]):
        #    plt.text(data.Price[i], y=data.LocationGrade[i], s=data.Real_Estate[i], alpha=0.8)

        # Quadrant Marker
        plt.text(x=2500, y=4.5, s="Q1", alpha=0.7, fontsize=14, color='b')
        plt.text(x=700, y=2, s="Q3", alpha=0.7, fontsize=14, color='b')
        plt.text(x=700, y=4.5, s="Q2", alpha=0.7, fontsize=14, color='b')
        plt.text(x=2500, y=2, s="Q4", alpha=0.7, fontsize=14, color='b')

        x = [1000, 700, 700, 700, 700, 700, 700, 1000, 2000, 1600, 3000, 1500, 1600, 1900, 2000]
        y = [4.8, 5, 5, 5, 5, 5, 5, 4, 4.8, 4, 5, 5, 5, 5, 5, 1, 1, 1, 1, 540, 550, 600, 650]

        artists = []
        for x0, y0, path in zip(x, y, paths):
            ab = AnnotationBbox(getImage(path), (x0, y0), frameon=False)
            artists.append(ax.add_artist(ab))

        plt.axhline(y=3.5, color='k', linestyle='--', linewidth=1)
        plt.axvline(x=1300, color='k', linestyle='--', linewidth=1)
        plt.show()
        st.pyplot(fig)


elif activites == "Comparision":
    realState = pd.Series(['Ayat', 'Noah', 'Gift'])
    life_ex = pd.Series([6, 5, 6])
    gni_pc = pd.Series([9, 8, 8])
    facility = pd.Series([8, 9, 8.5])
    hdi_df = pd.DataFrame({'realState': realState, 'life_ex': life_ex,
                           'gni_pc': gni_pc, 'Facility': facility})
    x_axis = st.selectbox('x_axis', data.columns)

    fig = px.scatter(hdi_df,
                     x='gni_pc',
                     y='life_ex',
                     # text="realState",
                     log_x=True,
                     size_max=60,
                     hover_name='realState',
                     color="realState",
                     title=f'life_ex vs. {x_axis}')

    # Country names
    fig.update_traces(textposition='top center')

    fig.update_layout(
        height=600,
        # title_text='GDP and Life Expectancy (Americas, 2007)'
    )

    for i in range(hdi_df.shape[0]):
        plt.text(hdi_df.gni_pc[i], y=hdi_df.life_ex[i], s=hdi_df.realState[i], alpha=0.8)

    st.plotly_chart(fig)

    plt.figure(figsize=(12, 8))

    fig, ax = plt.subplots()

    # Scatterplot
    ax = sns.scatterplot(data=hdi_df, x='gni_pc', y='life_ex')

    # Title
    plt.title(f"Real State Comparision")

    # x and y axis labels
    plt.xlabel("Price")
    plt.ylabel('Quality')

    # real state names
    for i in range(hdi_df.shape[0]):
        plt.text(hdi_df.gni_pc[i], y=hdi_df.life_ex[i], s=hdi_df.realState[i], alpha=0.8)

    # Quadrant Marker
    plt.text(x=8.8, y=5.2, s="Q4", alpha=0.7, fontsize=14, color='b')
    plt.text(x=8.2, y=5.2, s="Q3", alpha=0.7, fontsize=14, color='b')
    plt.text(x=8.2, y=5.8, s="Q2", alpha=0.7, fontsize=14, color='b')
    plt.text(x=8.8, y=5.8, s="Q1", alpha=0.7, fontsize=14, color='b')

    # display the logos
    paths = [
        '/Users/geletawsahle/Desktop/CP_2020_WEPAPP/JULogo.png',
        '/Users/geletawsahle/Desktop/CP_2020_WEPAPP/VUBLogo.png',
        '/Users/geletawsahle/Desktop/CP_2020_WEPAPP/VUBLogo.png',
    ]
    x = [8.1, 8.9, 8.1]
    y = [6, 6, 5]

    artists = []
    for x0, y0, path in zip(x, y, paths):
        ab = AnnotationBbox(getImage(path), (x0, y0), frameon=False)
        artists.append(ax.add_artist(ab))

    # Benchmark Mean values
    # plt.axhline(y=hdi_df.life_ex.mean(), color='k', linestyle='--', linewidth=1)
    # plt.axvline(x=hdi_df.gni_pc.mean(), color='k',linestyle='--', linewidth=1)

    plt.axhline(y=5.5, color='k', linestyle='--', linewidth=1)
    plt.axvline(x=8.5, color='k', linestyle='--', linewidth=1)
    plt.show()
    st.pyplot(fig)


elif activites == "Next Buyer":
    # pre-processing
    preprocessingType = ["Outlier", "NoisyValue", "Missing Value"]
    preprocessing = st.multiselect("Preprocessing", preprocessingType)
    # st.write(data["በሽያጭ_ሂደቱ_ያጋጠሙ_ችግሮች"].unique().tolist())
    # st.write(data.በሽያጭ_ሂደቱ_ያጋጠሙ_ችግሮች.unique())
    # st.write(Counter(data['በሽያጭ_ሂደቱ_ያጋጠሙ_ችግሮች']).items())
    xx = pd.DataFrame(data.ሌሎች_ሽያጭ_ሂደቱ_የረዱ_ጉዳዮች.value_counts().reset_index().values,
                      columns=["ሌሎች_ሽያጭ_ሂደቱ_የረዱ_ጉዳዮች", "Aggregate"])
    xx1 = xx.to_csv('/Users/geletawsahle/Desktop/CP_2020_WEPAPP/xx.csv')
    st.write(xx)


else:
    st.write("Choice the required activites to proceed")


chatbot = st.sidebar.checkbox("Enable Chatbot")

st.sidebar.subheader("@Copyright: 2021")
