import streamlit as st
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.special import kv
from io import BytesIO
import base64
import scipy.special as kt
from PIL import Image

def asinh(x):
    return np.arcsinh(x)
def newfunc(u, w):
    L1 = lambda y: kv(0, np.sqrt(w**2 + y**2))
    L, _ = quad(L1, 0, u)
    return L

def sph2cart(azimuth,elevation,r):
    x = r * np.cos(elevation) * np.cos(azimuth)
    y = r * np.cos(elevation) * np.sin(azimuth)
    z = r * np.sin(elevation)
    return x, y, z





######################################################## Define pages

def intro() :
    st.title("Horizontal Collector Well Simulator")
    st.divider()

    st.markdown("The following is a basic :red[app], that can be used to calculate the Drawdown for a Horizonatal collector well (:red[HCW].)")
    st.markdown(" The app calculations in the app are based using Hantush and Papadopulos (1962) where drawdown is calcualted as follows")
    image = Image.open('appeqn.png')
    st.image(image, caption='Drawdonwn acc. Hantush and Papadopulus 1962')

    
    
    
    """
The Authors of the following App are listed below
* Vibhu Batheja
* Prabhas K. Yadav
* Ashima
"""




########################################################################################

def datainput():
    st.header(" Please input the data on this page for the Well")
    st.divider()
    st.subheader("The aid of the following figures can be used for Data Input")

    image2 = Image.open('HYJO-D-21-00139_Fig_01.tif')
    

    image3=Image.open('HYJO-D-21-00139_Fig_08.tif')
    st.image(image2,width=600,caption='Figure 1')
    st.image(image3,width=600, caption="Figure 2")



 # Aquifer characteristics
    st.sidebar.subheader("Aquifer Characteristics")
    aquifer_type = st.sidebar.selectbox("Aquifer Type", ["Confined", "Unconfined"])
    if aquifer_type == "Confined" :
        aquifer_type=1
    if aquifer_type == "Unconfined" :
        aquifer_type=2
    B = st.sidebar.number_input("Thickness", value=35.66)
    Kv = st.sidebar.number_input("Vertical Hydraulic Conductivity", value=20)
    Kh = st.sidebar.number_input("Horizontal Hydraulic Conductivity", value=20)
    Ss = st.sidebar.number_input("Specific Storage (L^-1)", value=1e-5)
    Sy = st.sidebar.number_input("Specific Yield", value=0.3)

    # For a leaky aquifer
    #st.sidebar.subheader("Leaky Aquifer")
    Kdash = st.sidebar.number_input("Conductivity of Aquitard", value=20)
    Bdash = st.sidebar.number_input("Thickness of Aquitard", value=100)
    numM = st.sidebar.number_input("Number of Points for Calculation", value=20)
    nterms = st.sidebar.number_input("Number of Terms in Summation", value=10)
 
    
    # Angled well
    st.sidebar.subheader("Angled Well")
    degorrad = st.sidebar.selectbox("Select Angle measure type", ["Degree", "Radian"])
    angled_well = st.sidebar.selectbox("Well Type", ["Horizontal", "Angled"])
    if angled_well == "Horizontal" :
        angled_well=1
    if angled_well == "Angled" :
        angled_well=2
    #beta = st.sidebar.number_input("Elevation Angle (Radians)", value=0.0)

    # Well
    st.sidebar.subheader("Well")
    x0 = st.sidebar.number_input("X-coordinate of Well Position", value=300)
    y0 = st.sidebar.number_input("Y-coordinate of Well Position", value=450)
    z0 = st.sidebar.number_input("Height of Laterals", value=20)
    Q = st.sidebar.number_input("Well Pumping Rate", value=1000)
    st.sidebar.markdown(":green[Note:] for pumping rate input :red[-ve], value for pumping and :blue[+ve] value for infiltartion")
    n_l = st.sidebar.number_input("Number of Laterals", value=1)
    L = []
    LS = []
    angle = []
    beta=[]
    st.sidebar.subheader("Lateral Data Input")
    for i in range(n_l):
        L.append(st.sidebar.number_input(f"Length of Lateral (L)", value=500))
        LS.append(st.sidebar.number_input(f"Screened Length of Lateral (L)", value=300))
        if degorrad == "Radian" :
         angle.append(st.sidebar.number_input(f"Azimuth Angle of Lateral (Radians)", value=0.1))
         beta.append(st.sidebar.number_input(f"Beta Angle of well {i+1} (Radians)", value=0.1))
        if degorrad == "Degree" :
         dor_angle=(st.sidebar.number_input(f"Azimuth Angle of Lateral  (Degree)", value=5.72)*0.0174533)
         dor_beta=(st.sidebar.number_input(f"Beta Angle of well (Degree)", value=5.72)*0.0174533)
         dor_angle= dor_angle*0.0174533
         dor_beta=dor_beta*0.0174533
         angle.append(dor_angle)
         beta.append(dor_beta)
    r_c = st.sidebar.number_input("Radius of Caisson", value=0)
    t = st.sidebar.number_input("Time Since Start of Pumping", value=365)

    # Mesh properties
    st.sidebar.subheader("Mesh Properties")
    z = st.sidebar.number_input("Height of Observation (L)", value=5)
    xmin = st.sidebar.number_input("Minimum X-Position of Mesh", value=200)
    xmax = st.sidebar.number_input("Maximum X-Position of Mesh", value=800)
    ymin = st.sidebar.number_input("Minimum Y-Position of Mesh", value=300)
    ymax = st.sidebar.number_input("Maximum Y-Position of Mesh", value=900)
    size = st.sidebar.number_input("Size of Grid Square (L)", value=50)

    return (aquifer_type, B, Kv, Kh, Ss, Sy, Kdash, Bdash, numM,nterms, angled_well, beta, x0, y0, z0, Q, n_l,L,LS,angle,r_c,t,z,xmin,xmax,ymin,ymax,size)


#################################################################


def results(aquifer_type, B, Kv, Kh, Ss, Sy, Kdash, Bdash, numM,nterms, angled_well, beta, x0, y0, z0, Q, n_l,L,LS,angle,r_c,t,z,xmin,xmax,ymin,ymax,size) :
    
    progress_bar = st.progress(0)
    beta2=beta


#####################################


    x = np.arange(xmin, xmax+size, size)
    y = np.arange(ymin, ymax+size, size)
    x1,x2 = np.meshgrid(x, y)
    mat = np.zeros((len(y), len(x)))
    gesamt_absenkung = np.zeros((len(y), len(x)))
    ges2=np.zeros((int(x1.shape[0])*int(x1.shape[1])))
    s = np.zeros((len(y), len(x)))
    ges=[]
    leny=len(y)
    lenx=len(x)
 #print((type(x1)),x1,len(x1),len(x2[1]))
                                              #gesarr=np.array need to add an array 
    for j in range(n_l):
     #if(((j/n_l)*100)%5==0) :
     #print(((j/n_l)*100),'%')
        L_l = L[j]
        theta_i = angle[j]
        Beta = beta2[j]
     
        z = B - z
        z0 = B - z0
        probar=[]
        for i in range(x1.size):
            progress_bar.progress((j * x1.size + i + 1) / (n_l * x1.size))
            #if(int((i/x1.size)*100)%10==0) :
             #   if(int((i/x1.size)*100) not in probar) :
             
              #   st.write(int((i/x1.size)*100),'%')
               #  probar.append(int((i/x1.size)*100))
         # ------------------------- Hantush&Papadopolous --------------------------
            x = x1.flatten()[i]
            y = x2.flatten()[i]
         
            theta, r = np.arctan2(y-y0, x-x0), np.sqrt((x-x0)**2 + (y-y0)**2)
         #print(theta,theta_i)
            sigma = r * np.cos(theta-theta_i) - r_c - L_l
            beta = r * np.sin(theta-theta_i)
            alpha  = r * np.cos(theta-theta_i) - r_c
         
            if aquifer_type == 1:
                v_dash = Kh/Sy
            elif aquifer_type == 2:
                v_dash = Kh*B/Sy
        
            if t > 2.5*B**2/v_dash:
                u_1 = (alpha**2 + beta**2) / (4*v_dash*t)
                u_2 = (sigma**2 + beta**2) / (4*v_dash*t)
             #print(u_1,u_2)
             #print(math.log(u_1))
                if(u_1==0):
                 u_1=1
                 beta=1
                W_1 = -0.5772 - np.log(u_1) + u_1 - (u_1**2)/(2*kt.factorial(2)) + (u_1**3)/(3*kt.factorial(3)) - (u_1**4)/(4*kt.factorial(4))
                W_2 = -0.5772 - np.log(u_2) + u_2 - (u_2**2)/(2*kt.factorial(2)) + (u_2**3)/(3*kt.factorial(3)) - (u_2**4)/(4*kt.factorial(4))
            
                term_1 = alpha * W_1
                term_2 = sigma * W_2
                term_3 = 2 * L_l - 2*beta*(np.arctan(alpha/beta) - np.arctan(sigma/beta))
 

                if(u_1==0):
                 term_3=0
                 term_1=0
                 term_2=0
              
                if z == -1:
                    term_4 = 0
                else:
                   term_4 = 0
                   for n in range(1, 51):
                        term_4 += 1/n * (newfunc(n*math.pi*alpha/B, n*math.pi*beta/B) -
                                      newfunc(n*math.pi*sigma/B, n*math.pi*beta/B)) * \
                               np.cos(n*math.pi*z/B) * np.cos(n*math.pi*z0/B)
             
                mat = Q/L_l/4/math.pi/Kh/B * (term_1 - term_2 + term_3 + 4*B/math.pi * term_4)
        
            
            elif t <= B**2 / (20 * v_dash):
                w0 = z_i - z
                lambda0 = z_i + z
                thing = v_dash * t / B**2
             
                mat = Q / (4 * np.pi * K * L_l) * (asinh(alpha / np.sqrt(beta**2 + w0**2)) -
                    asinh(sigma / np.sqrt(beta**2 + w0**2)) + 
                 asinh(alpha / np.sqrt(beta**2 + lambda0**2)) - 
                 asinh(sigma / np.sqrt(beta**2 + lambda0**2)) -
                 2 * asinh(alpha / np.sqrt(beta**2 + (lambda0 + B*thing)**2)) +
                 2 * asinh(sigma / np.sqrt(beta**2 + (lambda0 + B*thing)**2)))
            else:
                print('Hantush not valid at this time')
 
        # print(gesamt_absenkung,gesamt_absenkung[i,j])
         #gesamt_absenkung[i,j]= gesamt_absenkung[i,j]+ mat
            ges.append(mat)
     
        ges2=np.array(ges)+ges2
        ges=[]
    
   

 #print(ges)
        

    gesamt_absenkung = np.array(ges2).reshape((int(x1.shape[0])),int(x1.shape[1]))



    fig, ax = plt.subplots(figsize=(8,6), subplot_kw=dict(aspect="equal"))

    # Create contour plot
    CS = ax.contour(x1, x2, gesamt_absenkung, linewidths=3, cmap='plasma',levels=10)
    plt.clabel(CS, inline=True, fontsize=10)

    # Plot the line
    for i in range(n_l):
        Ls=LS[i]
        L_l = L[i]
        theta_i = angle[i]
        beta = beta2[i]
        xx,yy,zz=sph2cart(theta_i,beta,Ls)
        XX, YY, ZZ = sph2cart(theta_i, beta, L_l)
        endx = x0 + XX
        endy = y0 + YY

        cx=x0+xx
        cy=y0+yy

        
        ax.plot([x0, endx], [y0, endy], color='k', linewidth=3, marker='o', markersize=10)
        ax.plot([x0, cx], [y0, cy], color='r', linewidth=3, marker='o', markersize=10)

    # Set labels and title
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Contour plot with line")

    # Show plot
    #st.pyplot(fig)

    buffer = BytesIO()
    fig.savefig(buffer, format='png')
    buffer.seek(0)


    data = base64.b64encode(buffer.read()).decode()

# Generate HTML code to display the image with width attribute
    html = f'<img src="data:image/png;base64,{data}" width=900, height=600>'

# Show the HTML code on Streamlit
    st.write(html, unsafe_allow_html=True)








    
st.set_page_config(page_title='', layout='wide')
# Add custom CSS to change the background color
st.markdown(
    """
    <style>
    body {
        background-color: ##00c69b;
    }
    </style>
    """,
    unsafe_allow_html=True
)



# Create a menu with options to navigate between pages








menu = ["Introduction","Data Input","Results"]
choice = st.sidebar.radio("Select a page", menu)

numM= None

# Show the appropriate page based on the user's choice
#if choice == "Home":
   # page_home()
if choice == "Introduction" :
  intro()

  
if choice == "Data Input":
   aquifer_type, B, Kv, Kh, Ss, Sy, Kdash, Bdash, numM,nterms, angled_well, beta, x0, y0, z0, Q, n_l,L,LS,angle,r_c,t,z,xmin,xmax,ymin,ymax,size=datainput()
   #st.write(aquifer_type)
   #st.write(aquifer_type, B, Kv, Kh, Ss, Sy, Kdash, Bdash, numM,nterms, angled_well, beta, x0, y0, z0, Q, n_l,L,LS,angle,r_c,t,z,xmin,xmax,ymin,ymax,size)
   if st.button('Compute Result'):
       choice="Results"
   if st.sidebar.button('Result')    :
       choice="Results"
    


####################################################  NEED TO try st.session_state check chatgpt

       

#st.write(beta)

if choice == "Results" :
    if  numM is None :
     aquifer_type, B, Kv, Kh, Ss, Sy, Kdash, Bdash, numM,nterms, angled_well, beta, x0, y0, z0, Q, n_l,L,LS,angle,r_c,t,z,xmin,xmax,ymin,ymax,size=datainput()
    #st.write(aquifer_type, B, Kv, Kh, Ss, Sy, Kdash, Bdash, numM,nterms, angled_well, beta, x0, y0, z0, Q, n_l,L,LS,angle,r_c,t,z,xmin,xmax,ymin,ymax,size) 
    #st.experimental_rerun()
    results(aquifer_type, B, Kv, Kh, Ss, Sy, Kdash, Bdash, numM,nterms, angled_well, beta, x0, y0, z0, Q, n_l,L,LS,angle,r_c,t,z,xmin,xmax,ymin,ymax,size)

 #   st.write(aquifer_type, B, Kv, Kh, Ss, Sy, Kdash, Bdash, numM,nterms, angled_well, beta, x0, y0, z0, Q, n_l,L,LS,angle,r_c,t,z,xmin,xmax,ymin,ymax,size)
 #  result(aquifer_type, B, Kv, Kh, Ss, Sy, Kdash, Bdash, numM,nterms, angled_well, beta, x0, y0, z0, Q, n_l,L,LS,angle,r_c,t,z,xmin,xmax,ymin,ymax,size)


    
