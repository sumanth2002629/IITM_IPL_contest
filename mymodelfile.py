# write your import here
from sklearn.dummy import DummyRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from xgboost import XGBRegressor
import pandas as pd
import numpy as np



class MyModel:

    def __init__(self):
        self.model=XGBRegressor()

    def fit(self, training_data):
        # players = ["YBK Jaiswal", "JC Buttler", "SV Samson", "D Padikkal", "SO Hetmyer", "R Ashwin", "R Parag", "TA Boult", "OC McCoy", "WP Saha", "Shubman Gill", "MS Wade", "HH Pandya", "DA Miller", "V Kohli", "F du Plessis", "RM Patidar", "GJ Maxwell", "MK Lomror", "KD Karthik", "Shahbaz Ahmed", "PWH de Silva", "HV Patel", "JR Hazlewood", "Q de Kock", "KL Rahul", "M Vohra", "DJ Hooda", "MP Stoinis", "E Lewis", "KH Pandya", "PVD Chameera", "PK Garg", "Abhishek Sharma", "RA Tripathi", "AK Markram", "N Pooran", "Washington Sundar", "R Shepherd", "J Suchith", "B Kumar", "Umran Malik", "JM Bairstow", "S Dhawan", "M Shahrukh Khan", "MA Agarwal", "LS Livingstone", "JM Sharma", "PN Mankad", "PP Shaw", "DA Warner", "MR Marsh", "RR Pant", "SN Khan", "R Powell", "AR Patel", "SN Thakur", "Kuldeep Yadav", "Ishan Kishan", "RG Sharma", "D Brevis", "Tilak Varma", "TH David", "Ramandeep Singh", "DR Sams", "RD Gaikwad", "DP Conway", "MM Ali", "N Jagadeesan", "AT Rayudu", "MS Dhoni", "Simarjeet Singh", "MJ Santner", "R Tewatia", "Rashid Khan", "VR Iyer", "N Rana", "A Tomar", "SS Iyer", "SW Billings", "AD Russell", "RK Singh", "SP Narine", "UT Yadav", "KS Williamson", "T Stubbs", "R Sanjay Yadav", "JJ Bumrah", "Lalit Yadav", "PBB Rajapaksa", "Harpreet Brar", "R Dhawan", "RD Chahar", "K Rabada", "Arshdeep Singh", "JDS Neesham", "A Badoni", "JO Holder", "Mohsin Khan", "Avesh Khan", "S Dube", "AM Rahane", "Shashank Singh", "M Jansen", "Mohammed Siraj", "RV Uthappa", "DJ Bravo", "M Theekshana", "Mukesh Choudhary", "HR Shokeen", "HE van der Dussen", "KS Bharat", "KS Sharma", "SP Jackson", "PJ Cummins", "TG Southee", "KA Pollard", "M Ashwin", "K Kartikeya", "RP Meredith", "RV Patel", "A Nortje", "KK Ahmed", "Kartik Tyagi", "Fazalhaq Farooqi", "B Indrajith", "AJ Finch", "AS Roy", "Shivam Mavi", "Harshit Rana", "SA Yadav", "B Sai Sudharsan", "Mandeep Singh", "SA Abbott", "S Gopal", "RA Jadeja", "D Pretorius", "PJ Sangwan", "LH Ferguson", "AS Joseph", "KK Nair", "DJ Mitchell", "A Manohar", "M Prasidh Krishna", "SS Prabhudessai", "MK Pandey", "JD Unadkat", "Anuj Rawat", "Yash Dayal", "NT Ellis", "VG Arora", "Ravi Bishnoi", "CV Varun", "V Shankar", "P Simran Singh", "OF Smith", "FA Allen", "TS Mills", "Aman Hakim Khan", "YS Chahal", "KR Sen", "Akash Deep", "K Gowtham", "Rasikh Salam", "CJ Jordan", "DJ Willey", "SE Rutherford", "Abdul Samad", "TL Seifert", "Mustafizur Rahman", "Navdeep Saini", "Anmolpreet Singh", "RA Bawa", "NM Coulter-Nile", "EJG Morgan", "Shakib Al Hasan", "AB de Villiers", "DT Christian", "TK Curran", "SK Raina", "SS Tiwary", "J Yadav", "SPD Smith", "CH Morris", "C Sakariya", "JJ Roy", "S Kaul", "KA Jamieson", "AF Milne", "CH Gayle", "KM Jadhav", "GD Phillips", "Sandeep Sharma", "SM Curran", "DL Chahar", "T Shamsi", "GHS Garton", "MC Henriques", "K Yadav", "PP Chawla", "Mohammad Nabi", "Sachin Baby", "DJ Malan", "Mohammed Shami", "Virat Singh", "KL Nagarkoti", "Harbhajan Singh", "Mujeeb Ur Rahman", "JA Richardson", "CR Woakes", "S Nadeem", "BA Stokes", "CA Lynn", "SP Goswami", "Gurkeerat Singh", "AT Carey", "JC Archer", "SS Cottrell", "T Banton", "M Vijay", "SR Watson", "P Dubey", "JR Philippe", "T Natarajan", "I Udana", "VR Aaron", "TU Deshpande", "Imran Tahir", "AS Rajpoot", "DW Steyn", "NS Naik", "AJ Tye", "JL Pattinson", "DS Kulkarni", "MJ McClenaghan", "C Munro", "KMA Paul", "A Mishra", "I Sharma", "MJ Guptill", "Basil Thampi", "YK Pathan", "PA Patel", "C de Grandhomme", "STR Binny", "IS Sodhi", "CA Ingram", "H Klaasen", "P Negi", "BB Sran", "AJ Turner", "DR Shorey", "CR Brathwaite", "AD Nath", "GC Viljoen", "Y Prithvi Raj", "KC Kariappa", "BCJ Cutting", "RK Bhui", "JL Denly", "SD Lad", "HF Gurney", "Yuvraj Singh", "MM Sharma", "GH Vihari", "S Lamichhane", "P Ray Barman", "M Markande", "JPR Scantlebury-Searles", "MK Tiwary", "AD Hales", "Anureet Singh", "DJM Short", "P Chopra", "JP Duminy", "NV Ojha", "BB McCullum", "LE Plunkett", "MG Johnson", "CJ Anderson", "G Gambhir", "B Laughlin", "A Dananjaya", "B Stanlake", "R Vinay Kumar", "MA Wood", "LMP Simmons", "KV Sharma", "IR Jaggi", "Bipul Sharma", "SL Malinga", "Vishnu Vinod", "TM Head", "MN Samuels", "Z Khan", "SE Marsh", "Swapnil Singh", "DR Smith", "JP Faulkner", "Ankit Soni", "P Kumar", "MM Patel", "HM Amla", "S Aravind", "S Badree", "A Choudhary", "IK Pathan", "AR Bawne", "DM Bravo", "AP Tare", "AD Mathews", "Ankit Sharma", "R Bhatia", "A Zampa", "AB Dinda", "Iqbal Abdulla", "ER Dwivedi", "R Sathish", "SB Jakati", "F Behardien", "KJ Abbott", "UT Khawaja", "GJ Bailey", "NLTC Perera", "M Morkel", "PV Tambe", "S Kaushik", "UBT Chand", "A Nehra", "PSP Handscomb", "A Ashish Reddy", "Parvez Rasool", "JA Morkel", "KP Pietersen", "KW Richardson", "P Sahu", "D Wiese", "MEK Hussey", "MA Starc", "Azhar Mahmood", "BE Hendricks", "RS Bopara", "J Botha", "GB Hogg", "Karanveer Singh", "RN ten Doeschate", "NJ Maddinson", "V Sehwag", "J Theron", "DJ Muthuswami", "MS Bisla", "Noor Ahmad", "RR Rossouw", "AN Ahmed", "RG More", "DJG Sammy", "CM Gautam", "DJ Hussey", "PP Ojha", "BJ Hodge", "Y Venugopal Rao", "S Anirudha", "YV Takawale", "VH Zol", "S Rana", "KK Cooper", "VS Malik", "M de Lange", "Shivam Sharma", "WD Parnell", "LRPL Taylor", "R Shukla", "DH Yagnik", "M Manhas", "LR Shukla", "L Balaji", "JH Kallis", "BR Dunk", "CA Pujara", "R Rampaul", "AM Nayar", "S Badrinath", "R Dravid", "CL White", "BB Samantray", "P Dogra", "LJ Wright", "RV Gomez", "DPMD Jayawardene", "B Chipli", "BJ Rohrer", "AC Gilchrist", "SK Trivedi", "RE van der Merwe", "DB Das", "SMSM Senanayake", "SR Tendulkar", "MC Juneja", "AB Agarkar", "LA Pomersbach", "A Mithun", "M Kartik", "Harmeet Singh", "P Awana", "UA Birla", "M Rawat", "K Upadhyay", "MS Gony", "KC Sangakkara", "R McLaren", "PA Reddy", "BMAJ Mendis", "OA Shah", "TL Suman", "BAW Mendis", "R Sharma", "S Narwal", "B Lee", "A Mukund", "TM Dilshan", "KB Arun Karthik", "J Syed Mohammad", "X Thalaivan Sargunam", "AG Murtaza", "IC Pandey", "A Chandila", "S Sreesanth", "SW Tait", "PC Valthaty", "AD Mascarenhas", "RT Ponting", "AB McDonald", "RJ Harris", "DB Ravi Teja", "MDKJ Perera", "AL Menaria", "Sunny Gupta", "JEC Franklin", "RP Singh", "M Muralitharan", "P Parameswaran", "JD Ryder", "MJ Clarke", "SC Ganguly", "AP Majumdar", "CJ Ferguson", "Harpreet Singh", "SD Chitnis", "N Saini", "HH Gibbs", "Y Nagar", "MD Mishra", "AC Thomas", "DJ Harris", "AA Jhunjhunwala", "AUK Pathan", "RJ Peterson", "AA Chavan", "V Pratap Singh", "RE Levi", "DL Vettori", "KMDN Kulasekara", "AC Blizzard", "DJ Jacobs", "CJ McKay", "M Kaif", "BA Bhatt", "RR Bhatkal", "A Singh", "DAJ Bracewell", "DE Bollinger", "Y Gnaneswara Rao", "JJ van der Wath", "CK Langeveldt", "S Sohal", "FY Fazal", "PR Shah", "JDP Oram", "SK Warne", "ND Doshi", "S Sriram", "JR Hopes", "TR Birt", "MJ Lumb", "A Symonds", "MV Boucher", "AS Raut", "SJ Srivastava", "M Klinger", "RR Powar", "Sunny Singh", "L Ablish", "I Malhotra", "GC Smith", "SB Wagh", "NL McCullum", "B Akhil", "TD Paine", "JE Taylor", "VVS Laxman", "BJ Haddin", "Joginder Sharma", "S Randiv", "SA Asnodkar", "AG Paunikar",
                #    "R Ninan", "NJ Rimmington", "SB Styris", "ML Hayden", "CRD Fernando", "Anirudh Singh", "B Sumanth", "A Kumble", "WPUJC Vaas", "PD Collingwood", "AC Voges", "Pankaj Singh", "C Madan", "KP Appanna", "ST Jayasuriya", "AP Dole", "MF Maharoof", "AB Barath", "T Thushara", "RS Sodhi", "AA Bilakhia", "Jaskaran Singh", "K Goel", "KAJ Roach", "S Ladda", "DP Nannes", "JM Kemp", "DR Martyn", "RS Gavaskar", "SE Bond", "A Uniyal", "S Tyagi", "YA Abdulla", "Mohammad Ashraful", "SM Katich", "WA Mota", "RJ Quiney", "NK Patel", "T Henderson", "Yashpal Singh", "SS Shaikh", "Mashrafe Mortaza", "C Nanda", "Shoaib Ahmed", "AN Ghosh", "LA Carseldine", "MN van Wyk", "SM Harwood", "L Ronchi", "RR Raje", "W Jaffer", "D du Preez", "R Bishnoi", "TM Srivastava", "GR Napier", "FH Edwards", "SB Bangar", "A Flintoff", "T Kohli", "A Chopra", "Kamran Khan", "S Vidyut", "CK Kapugedera", "Kamran Akmal", "Sohail Tanvir", "Mohammad Asif", "VY Mahesh", "GD McGrath", "AS Yadav", "Shahid Afridi", "SP Fleming", "SM Pollock", "Salman Butt", "Umar Gul", "Misbah-ul-Haq", "LPC Silva", "M Ntini", "VS Yeligati", "Mohammad Hafeez", "DJ Thornely", "H Das", "DNT Zoysa", "J Arunkumar", "DT Patil", "Abdur Razzak", "Shoaib Malik", "Shoaib Akhtar", "PM Sarvesh Kumar", "DP Vijaykumar", "Younis Khan", "D Salunkhe", "T Taibu", "RR Sarwan", "VRV Singh", "U Kaul", "S Chanderpaul", "D Kalyankrishna", "MA Khote", "SB Joshi", "DS Lehmann", "AA Noffke", "R Sai Kishore", "M Pathirana", "PH Solanki", "DG Nalkande", "IC Porel", "AU Rashid", "S Sandeep Warrier", "Akash Singh", "KM Asif", "L Ngidi", "LI Meriwala", "Jalaj S Saxena", "Monu Kumar", "CJ Green", "K Khejroliya", "O Thomas", "JP Behrendorff", "SC Kuggeleijn", "S Midhun", "CJ Dala", "MJ Henry", "NB Singh", "SS Agarwal", "Tejas Baroka", "SM Boland", "JW Hastings", "GS Sandhu", "P Suyal", "K Santokie", "BW Hilfenhaus", "Anand Rajan", "MG Neser", "TP Sudhindra", "RW Price", "SS Mundhe", "P Prasanth", "AM Salvi", "AA Kazi", "MB Parmar", "C Ganapathy", "RA Shaikh", "SS Sarkar", "RR Bose", "B Geeves", "A Nel", "Gagandeep Singh", "P Amarnath", "Harry Brook", "Joe Root", "Cameron Green", "Sikandar Raza", "Litton Das", "Kusal Mendis", "Phil Salt", "Reece Topley", "Akeal Hosein", "Shubham Khajuria", "Rohan Kunnummal", "Chethan L.R.", "Shaik Rasheed", "Saurabh Kumar", "Vivrant Sharma", "Nishant Sindhu", "Sanvir Singh", "Samarth Vyas", "Dinesh Bana", "Abhimanyu Easwaran", "Sumit Kumar", "Upendra Singh Yadav", "Mukesh Kumar", "Lance Morris", "Yash Thakur", "Mujtaba Yousuf", "Chintal Gandhi", "Izharulhuq Naveed", "Himanshu Sharma", "Subhranshu Senapati", "Will Jacks", "Paul Stirling", "Dasun Shanaka", "Taskin Ahmed", "Blessing Muzarabani", "Johnson Charles", "Andre Fletcher", "Shai Hope", "Tom Latham", "Ben Mcdermott", "Lorcan Tucker", "Pukhraj Mann", "Akshat Raghuwanshi", "Himanshu Rana", "Shoun Roger", "Will Smeed", "Manoj Bhandage", "Gerald Coetzee", "Duan Jansen", "Evan Jones", "Abid Mushtaq", "Suryansh Shedge", "Akash Vashisht", "Donovan Ferreira", "Urvil Patel", "Kirant Shinde", "Vishnu Solanki", "Vidwath Kaverappa", "Rajan Kumar", "Ravi Kumar", "Arzan Nagwaswalla", "Akash Singh", "Paul Van Meekeren", "Vyshak Vijay Kumar", "S.Ajith Ram", "Satyajeet Bachhav", "Yuvraj Chudasama", "Peter Hatzoglou", "Karthik Meiyappan", "Suyash Sharma", "Shivam Sharma", "Reeza Hendricks", "Christiaan Jonker", "Brandon King", "Pathum Nissaanka", "Harry Tector", "Najibullah Zadran", "Qais Ahmad", "Charith Asalanka", "Michael Bracewell", "Jamie Overton", "Richard Gleeson", "Naveen Ul Haq", "Lahiru Kumara", "Joshua Little", "Dilshan Madushanka", "Luke Wood", "Priyansh Arya", "Matthew Breetzke", "Shivam Chauhan", "Rahul Gahlaut", "Sudip Gharami", "Amandeep Khare", "Bhanu Pania", "Ekant Sen", "Himanshu Bisht", "Mickil Jaiswal", "G.Aniketh Reddy", "Atit Sheth", "Tanay Thyagarajann", "Sumeet Verma", "Ajitesh Guruswamy", "Yash Kothari", "Suresh Kumar", "Kumar Kushagra", "Anmol Malhotra", "Robin Minz", "Agniv Pan", "Priyesh Patel", "Mitesh Patel", "Abishek Porel", "Nitish Kumar Reddy", "Bharat Sharma", "Vivek Singh", "Basit Bashir", "Nandre Burger", "Sakib Hussain", "Waseem Khanday", "Ravi Kiran Majeti", "Anuj Raj", "Avinash Singh", "Prince Yadav", "Mushtaq Beg", "Rocky Bhasker", "Sanjith Devaraj", "Raghav Goyal", "Allah Mohammad", "Lalit Mohan", "Bhuwan Rohilla", "Aman Sharma", "Manav Suthar", "Afif Hossain", "Sisanda Magala", "Craig Overton", "Dhananjaya Silva", "Dunith Wellalage", "Daryn Dupavillon", "David Payne", "Glenton Stuurman", "Anirudh Balachander", "Gourav Choudhary", "Saurav Chuahan", "Kumar Deobrat", "Chirag Gandhi", "Madhav Kaushik", "Priyank Panchal", "Ayush Pandey", "Rohan Patil", "Sanjay Ramaswamy", "Siddharth Yadav", "Rehan Ahmed", "Vaisakh Chandran", "Harsh Dubey", "Tanush Kotian", "Ninad Rathva", "B. Surya", "Jordan Thompson", "Shivank Vashisth", "Christopher Benjamin", "Connor Esterhuizen", "Mohd Arslan Khan", "Mamidi Krishna", "Fazil Makaya", "Kunal Rathore", "Ateev Saini", "Bipin Saurabh", "B.R. Sharath", "Yashovardhan Singh", "Lakshay Thareja", "Mohit Avasthi", "Ottneil Baartman", "Gurnoor Singh Brar", "Shahrukh Dar", "Thomas Helm", "Venkatesh Muralidhara", "Geet Puri", "E. Sanketh", "Ajay Sarkar", "Ashok Sharma", "Kanwar Singh", "Roston Chase", "Rahkeem Cornwall", "Karim Janat", "Keshav Maharaj", "Shivam Chaudhary", "Ashwin Das", "James Fuller", "Chirag Jani", "Akshay Karnewar", "Bhagmender Lather", "Lone Muzaffar", "Pulkit Narang", "Rohit Rayudu", "Sameer Rizvi", "Tunish Sawkar", "Sonu Yadav", "Auqib Dar", "Mukhtar Hussain", "Ashwani Kumar", "Hemant Kumar", "Nathan McAndrew", "Rajesh Mohanty", "Ravi Sharma", "Vikash Singh", "Ruben Trumpelmann", "Koushik Vasuki", "Vasu Vats", "Shubham Agrawal", "Anshul Kamboj", "Azim Kazi", "Dev Lakra", "Ajay Mandal", "Abdul P A", "Jitender Pal", "Ritwik Roy Chowdhury", "Shubham Singh", "Avneesh Sudha", "Asad Jamil Ahmed", "Aashish Bhatt", "McKenny Clarke", "Shubham Kapse", "Gourav Koul", "Raunak Kumar", "Trilok Nag", "Atal Bihari Rai", "Ramon Simmonds", "Rajeev Singh", "Mohd. Wasim", "Atharva Ankolekar", "Khizar Dafedar", "Naman Dhir", "Sahil Dhiwan", "Sampark Gupta", "Jordan Hermann", "Hayden Kerr", "Salman Khan", "Sairaj Patil", "Divyaansh Saxena", "Purnank Tyagi", "Deepraj Gaonkar", "Deepesh Nailwal", "Arjun Rapria", "Shashwat Rawat", "Sumit Ruikar", "Rajandeep Singh", "Anunay Singh", "Digvesh Singh", "Pranshu Vijayran", "Prerit Dutta", "Ramakrishna Ghosh", "Shubhang Hegde", "Shamshuzama Kazi", "Ayaz Khan", "Amit Pachhara", "Akul Pandove", "Mohit Rathee", "Garv Sangwan", "Shubham Sharma", "Nehal Wadhera", "Amit Yadav", "Amit Ali", "Rishabh Chauhan", "Matthew Forde", "Sammar Gajjar", "Rajneesh Gurbani", "Divyansh Joshi", "Dhruv Patel", "Jack Prestwidge", "Aditya Sarvate", "Mayank Dagar", "Sagar Solanki", "Prenelan Subrayen", "Bhagath Verma", "Rajvardhan Hangargekar", "Mayank Yadav", "Rahmanullah Gurbaz", "Kyle Mayers", "Yudhvir Singh", "Arjun Tendulkar", "Arshad Khan", "Shams Mulani", "Matthew Short", "Baltej Singh", "Atharva Taide", "Shivam Singh", "Dhruv Jurel"]

        data = training_data[0]

        ids = data['ID'].unique()


        new_data = []
        for id in ids:
            match = data[data['ID'] == id]
            match = match[match['overs'] < 6]
            innings1 = match[match['innings'] == 1].reset_index(drop=True)
            innings2 = match[match['innings'] == 2].reset_index(drop=True)

            if not innings1.empty:
                batsmen1 = innings1['batter'].unique()
                bowlers1 = innings1['bowler'].unique()

                all_batsmen1 = batsmen1[0]
                for b in batsmen1[1:]:
                    all_batsmen1 += (", "+b)

                all_bowlers1 = bowlers1[0]
                for b in bowlers1[1:]:
                    all_bowlers1 += (", "+b)
                new_data.append([1, innings1['BattingTeam'][0] if not innings1.empty else 'NA', innings2['BattingTeam']
                                [0] if not innings2.empty else 'NA', all_batsmen1, all_bowlers1, innings1['total_run'].sum()])

            if not innings2.empty:
                bowlers2 = innings2['bowler'].unique()
                batsmen2 = innings2['batter'].unique()

                all_batsmen2 = batsmen2[0]
                for b in batsmen2[1:]:
                    all_batsmen2 += (", "+b)

                all_bowlers2 = bowlers2[0]
                for b in bowlers2[1:]:
                    all_bowlers2 += (", "+b)
                new_data.append([2, innings2['BattingTeam'][0] if not innings2.empty else 'NA', innings1['BattingTeam']
                                [0] if not innings1.empty else 'NA', all_batsmen2, all_bowlers2, innings2['total_run'].sum()])
                
        train = pd.DataFrame(new_data,columns=["innings","batting_team","bowling_team","batsmen","bowlers","runs"])
                
        #initialising all players to perform onehot encoding
        # for player in players:
        #     train[player[1][0]] = 0

        # row_index = 0

        #updating batsmen and bowlers of every innings
        # for innings in train.iterrows():
        #     batsmen = innings[1][3].split(", ")
        #     bowlers = innings[1][4].split(", ")
            
        #     for batsman in batsmen:
        #         train.loc[row_index,[batsman]] = 1
            
        #     row_index+=1

        train = pd.get_dummies(train, columns = ['batting_team', 'bowling_team'])
        train.drop(axis=1,columns=['batsmen','bowlers'], inplace=True)
        train_x=train.drop(axis=1,columns=['runs']).to_numpy()
        train_y=train['runs'].to_numpy()

        x_train, x_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.2, random_state=42)

        self.model.fit(train_x,train_y)
        cv = RepeatedKFold(n_splits=3, n_repeats=5, random_state=1)
        scores = cross_val_score(self.model, train_x, train_y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
        print('Mean MAE: %.3f (%.3f)' % (scores.mean(), scores.std()))
        return self

    def predict(self, test_data):
        data = {'innings':[], 'batting_team_Chennai Super Kings':[],
       'batting_team_Deccan Chargers':[], 'batting_team_Delhi Capitals':[],
       'batting_team_Delhi Daredevils':[], 'batting_team_Gujarat Lions':[],
       'batting_team_Gujarat Titans':[], 'batting_team_Kings XI Punjab':[],
       'batting_team_Kochi Tuskers Kerala':[],
       'batting_team_Kolkata Knight Riders':[],
       'batting_team_Lucknow Super Giants':[], 'batting_team_Mumbai Indians':[],
       'batting_team_Pune Warriors':[], 'batting_team_Punjab Kings':[],
       'batting_team_Rajasthan Royals':[], 'batting_team_Rising Pune Supergiant':[],
       'batting_team_Rising Pune Supergiants':[],
       'batting_team_Royal Challengers Bangalore':[],
       'batting_team_Sunrisers Hyderabad':[], 'bowling_team_Chennai Super Kings':[],
       'bowling_team_Deccan Chargers':[], 'bowling_team_Delhi Capitals':[],
       'bowling_team_Delhi Daredevils':[], 'bowling_team_Gujarat Lions':[],
       'bowling_team_Gujarat Titans':[], 'bowling_team_Kings XI Punjab':[],
       'bowling_team_Kochi Tuskers Kerala':[],
       'bowling_team_Kolkata Knight Riders':[],
       'bowling_team_Lucknow Super Giants':[], 'bowling_team_Mumbai Indians':[],
       'bowling_team_NA':[], 'bowling_team_Pune Warriors':[],
       'bowling_team_Punjab Kings':[], 'bowling_team_Rajasthan Royals':[],
       'bowling_team_Rising Pune Supergiant':[],
       'bowling_team_Rising Pune Supergiants':[],
       'bowling_team_Royal Challengers Bangalore':[],
       'bowling_team_Sunrisers Hyderabad':[]}
        
        tes_data = pd.DataFrame(data)
        
        X_test = test_data

        batting_team_1 = test_data["batting_team"][0]
        batting_team_2 = test_data["batting_team"][1]
        bowling_team_1 = test_data["bowling_team"][0]
        bowling_team_2 = test_data["bowling_team"][1]

        tes_data.loc[0] = [0 for i in range(38)]
        tes_data.loc[len(tes_data.index)] = [1] +  [0 for i in range(37)]

        tes_data.at[0, "batting_team_"+batting_team_1] = 1
        tes_data.at[0, "bowling_team_"+bowling_team_1] = 1
        tes_data.at[1, "batting_team_"+batting_team_2] = 1
        tes_data.at[1, "bowling_team_"+bowling_team_2] = 1

        # print(batting_team_1, bowling_team_1,batting_team_2, bowling_team_2)
        return self.model.predict(tes_data)
