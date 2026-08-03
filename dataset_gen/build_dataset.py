"""
Generates a diverse, topic-matched dataset/ai_vs_human.csv.

Why this exists:
The original dataset had only 108 rows built from 15 human anecdotes and 15
AI paragraphs about entirely different (unrelated) topics. That let the model
learn TOPIC (fishing trip vs. quantum computing) instead of STYLE (human vs.
AI phrasing), and the vocabulary was so narrow that any real-world input text
shared almost no TF-IDF features with training data, so the model always fell
back to predicting "Human" with high confidence, regardless of input.

This script builds many more base paragraphs (~50 topics), each written once
in a natural human voice and once in a typical LLM-generated voice, so the
model is forced to learn stylistic signal instead of topic vocabulary. It
then augments by concatenating same-class paragraphs from different topics,
producing a larger, lexically diverse dataset that generalizes.
"""
import os
import random
import pandas as pd

random.seed(42)

# Each tuple: (topic, human_paragraph, ai_paragraph)
PAIRS = [
("morning coffee",
 "I usually stumble into the kitchen half asleep and just want the coffee maker to hurry up. Some mornings I forget to add water and only realize when it starts hissing weird. Not glamorous, just how I get through the day.",
 "Establishing a consistent morning coffee routine can significantly enhance productivity and overall well-being. It is important to note that the quality of the beans, brewing temperature, and grind size all play a crucial role in the final flavor profile. Ultimately, a mindful approach to this daily ritual can set a positive tone for the rest of the day."),

("weekend hiking",
 "We got lost for like twenty minutes on the trail because someone (me) misread the map app. Ended up finding a tiny waterfall we weren't even looking for, so honestly worth it. My legs are going to hate me tomorrow though.",
 "Hiking on weekends offers numerous benefits, including improved cardiovascular health and reduced stress levels. Furthermore, spending time in nature has been shown to boost mental clarity and foster a sense of calm. It is essential to plan routes carefully and carry adequate supplies to ensure a safe and enjoyable experience."),

("used car shopping",
 "Went to look at a used hatchback yesterday and the guy wouldn't budge on price even though the AC barely worked. Test drove it anyway just to see. Might just keep saving and look again next month.",
 "When purchasing a used vehicle, it is paramount to conduct thorough research and inspect the car's maintenance history. Consequently, buyers should prioritize a professional mechanical inspection before finalizing any transaction. This diligent approach helps mitigate the risk of costly repairs down the line."),

("cooking dinner",
 "Tried to make stir fry with whatever was left in the fridge and it turned out way better than expected. The trick was just not overcrowding the pan, something my mom always nags me about. Ate the leftovers cold standing at the counter, no shame.",
 "Preparing a well-balanced dinner involves careful consideration of nutritional value and ingredient freshness. Moreover, incorporating a variety of vegetables and lean proteins can contribute to a healthier lifestyle. In summary, thoughtful meal planning fosters both culinary satisfaction and long-term wellness."),

("remote work",
 "Working from home means I roll out of bed and I'm basically at my desk in two minutes, which is nice, but my back is destroyed from this dining chair. Also I talk to my cat more than actual coworkers some days.",
 "Remote work has fundamentally transformed the modern professional landscape, offering employees increased flexibility and autonomy. Nevertheless, it is crucial to establish clear boundaries between work and personal life to maintain productivity. Organizations that embrace hybrid models often report higher employee satisfaction and retention."),

("garden project",
 "Planted tomatoes way too early this year and a late frost pretty much wiped half of them out. Replanted the survivors closer to the fence where it's warmer. Fingers crossed for a decent harvest this time.",
 "Establishing a home garden requires careful attention to soil composition, sunlight exposure, and seasonal timing. Additionally, selecting climate-appropriate plant varieties is paramount to achieving a successful yield. With consistent care, a garden can become both a rewarding and sustainable hobby."),

("smartphone upgrade",
 "My phone battery dies by like 2pm now so I finally caved and ordered a new one. Kind of annoyed I have to redo all my app logins again honestly. At least the camera is supposed to be way better.",
 "Upgrading to a new smartphone can substantially enhance user experience through improved processing power and camera capabilities. It is worth considering that battery degradation over time is a primary driver behind most upgrade decisions. Ultimately, consumers should weigh cost against genuine feature improvements before making a purchase."),

("job interview",
 "I completely blanked on a question about my weaknesses and just kind of rambled for a bit. Pretty sure I recovered by the end but who knows. Now it's just the waiting game checking my email every five minutes.",
 "Preparing thoroughly for a job interview is paramount to making a favorable impression on potential employers. Furthermore, candidates should research the company culture and anticipate common behavioral questions in advance. This proactive approach underscores professionalism and genuine interest in the role."),

("new puppy",
 "The puppy chewed through another shoe last night, my third pair gone this month. He's ridiculously cute about it though, just sits there looking proud of himself. Guess I'm buying a crate today.",
 "Welcoming a new puppy into the household necessitates a structured routine encompassing training, nutrition, and socialization. Consequently, early exposure to varied environments fosters a well-adjusted and confident adult dog. Consistent positive reinforcement remains a cornerstone of effective puppy training."),

("grocery shopping",
 "Forgot my list again and just wandered the aisles hoping stuff would jog my memory. Ended up buying three kinds of cheese and no actual dinner ingredients. Classic me, honestly.",
 "Effective grocery shopping strategies can significantly reduce both food waste and unnecessary expenditure. Moreover, planning meals in advance and adhering to a prepared list fosters more intentional purchasing decisions. This methodical approach ultimately supports healthier eating habits and better budget management."),

("public transit commute",
 "The bus was fifteen minutes late again and packed shoulder to shoulder by the time it showed up. Stood the whole way holding onto a pole for dear life. At this point I should just budget the delay into my morning.",
 "Public transit systems play a pivotal role in reducing urban congestion and promoting environmental sustainability. Nevertheless, reliability and punctuality remain persistent challenges that municipalities must address. Investment in modern infrastructure is essential to enhancing the overall commuter experience."),

("home renovation",
 "We ripped up the old carpet ourselves to save money and found some seriously ugly flooring underneath. Now we're stuck deciding between tile and hardwood while living with plywood subfloor for a bit. Weekends are basically just Home Depot runs now.",
 "Undertaking a home renovation project requires meticulous planning and a realistic budget to avoid unforeseen expenses. Furthermore, homeowners should prioritize structural repairs before pursuing purely aesthetic upgrades. A phased approach can help manage both costs and disruption to daily living."),

("learning guitar",
 "My fingers are still killing me from practicing chord changes for like an hour straight. I can almost play the intro to that one song without stopping, which feels like a huge win. Calluses are apparently just part of the deal.",
 "Learning to play the guitar involves developing muscle memory through consistent and deliberate practice. Additionally, mastering fundamental chords and transitions lays the groundwork for more advanced techniques. Patience and regular repetition are crucial to achieving long-term proficiency."),

("streaming services",
 "Cancelled one subscription and immediately regretted it because obviously the one show I wanted to rewatch left too. Now I'm juggling three different apps just to keep up with everything. Feels like cable all over again honestly.",
 "The proliferation of streaming services has significantly diversified entertainment consumption habits among modern audiences. However, subscription fatigue has emerged as a notable concern, prompting consumers to reevaluate their media expenditures. Content bundling may represent a viable solution to this ongoing industry challenge."),

("running a 5k",
 "Barely finished the last mile without stopping to walk, my lungs were on fire. Beat my time from last month by like ninety seconds though, so I'll take it. Already sore in muscles I forgot existed.",
 "Training for a five-kilometer race requires a structured regimen that gradually builds cardiovascular endurance. Moreover, incorporating rest days and proper nutrition is paramount to preventing injury and optimizing performance. Consistency over several weeks typically yields measurable improvements in pace and stamina."),

("family gathering",
 "My uncle told the same fishing story for the third year in a row and somehow it still got a laugh. Grandma's pie disappeared within like ten minutes flat. Chaotic but honestly kind of nice to have everyone together.",
 "Family gatherings serve a pivotal role in strengthening interpersonal bonds and preserving shared traditions across generations. Furthermore, these occasions often provide an opportunity for open communication and mutual support. Cultivating such connections contributes meaningfully to overall emotional well-being."),

("budgeting app",
 "Downloaded a budgeting app after realizing I had no idea where my paycheck kept disappearing to. Turns out it's mostly takeout, which, yeah, no surprise there. Trying the whole envelope method now but digitally.",
 "Utilizing a budgeting application can substantially improve financial literacy and spending discipline among users. Consequently, real-time expense tracking allows individuals to identify and curtail unnecessary discretionary spending. This data-driven approach fosters more informed and sustainable financial decision-making."),

("office meeting",
 "The meeting could have been an email, as usual, and ran twenty minutes over because two people kept going back and forth. I mostly just doodled in my notebook and nodded at the right times. Same story every Tuesday.",
 "Efficient meeting management is paramount to preserving organizational productivity and employee engagement. Furthermore, establishing a clear agenda beforehand can significantly reduce time spent on tangential discussions. Organizations that prioritize concise, purposeful meetings often see improved overall workflow."),

("baking bread",
 "The dough didn't rise nearly as much as I hoped, probably because my kitchen is freezing this time of year. Still tasted decent toasted with butter, just denser than usual. Next time I'm proofing it near the oven instead.",
 "Baking bread from scratch involves a delicate balance of ingredient ratios, fermentation time, and ambient temperature. Moreover, understanding the role of gluten development is crucial to achieving the desired texture and rise. Patience throughout the proofing process ultimately determines the quality of the final loaf."),

("used bookstore visit",
 "Spent way too long in the dusty back corner of the bookstore and walked out with four books I definitely don't have time to read. The smell of old paper gets me every time. Worth every dollar of the twelve bucks I spent.",
 "Independent used bookstores continue to hold cultural significance despite the rise of digital reading platforms. Furthermore, browsing physical shelves often fosters serendipitous discoveries that algorithmic recommendations cannot replicate. Supporting these establishments helps preserve a vital component of local literary communities."),

("car maintenance",
 "The mechanic called with the bad news that it's not just the brakes, it's the rotors too. Wallet is going to feel that one for a while. At least the weird grinding noise should finally stop.",
 "Regular vehicle maintenance is paramount to ensuring both longevity and safety on the road. Consequently, addressing minor issues promptly can prevent more costly repairs from developing over time. Scheduling routine inspections remains one of the most effective strategies for responsible car ownership."),

("classroom teaching",
 "Half the class was clearly checked out by the last period on a Friday, staring at the clock more than the whiteboard. Tried a quick group activity just to get some energy back in the room. Small wins, I guess.",
 "Effective classroom instruction requires educators to employ varied pedagogical strategies to sustain student engagement. Moreover, incorporating interactive activities can significantly enhance information retention among learners. Continuous professional development remains essential for teachers navigating evolving educational demands."),

("apartment hunting",
 "Toured a place today that looked great in photos but the kitchen was somehow smaller than my old dorm room. Landlord also dodged my question about the noisy upstairs neighbors. Back to scrolling listings tonight.",
 "Searching for a suitable apartment necessitates careful evaluation of location, budget, and lease terms. Furthermore, prospective tenants should thoroughly inspect the property and clarify any ambiguous clauses before signing. This diligent approach helps prevent unforeseen complications throughout the tenancy."),

("photography hobby",
 "Finally got a shot of the sunset I've been trying to capture for weeks, colors were unreal that evening. Still learning how to not blow out the highlights though. Editing takes way longer than actually taking the picture.",
 "Photography as a hobby offers a rewarding creative outlet while fostering a deeper appreciation for visual composition. Additionally, mastering technical elements such as exposure and aperture is paramount to producing compelling images. Consistent practice and experimentation remain key to developing a distinctive artistic style."),

("online shopping return",
 "The sweater I ordered looked nothing like the photos, more of a weird mustard yellow than the mustard the site showed. Printing the return label took longer than actually deciding to send it back. Store credit it is, I guess.",
 "Navigating online retail returns requires consumers to understand each platform's specific policies and timelines. Moreover, retaining original packaging and documentation can streamline the refund or exchange process considerably. Transparent return policies ultimately foster greater consumer trust in e-commerce platforms."),

("neighborhood block party",
 "Someone's speaker cut out mid-song and there was a solid two minutes of everyone just standing around awkwardly. The kids ran through a sprinkler the whole afternoon, completely unbothered by any of it. Good excuse to finally meet the new neighbors.",
 "Community block parties play a pivotal role in strengthening neighborhood cohesion and fostering a sense of belonging. Furthermore, such events provide residents with valuable opportunities to build lasting relationships. Local organizers often find that consistent, well-planned gatherings enhance overall community resilience."),

("studying for exams",
 "Pulled an all-nighter cramming for the exam and honestly can't remember half of what I read by morning. Coffee number four is doing absolutely nothing at this point. Just hoping enough of it sticks for the multiple choice section.",
 "Effective exam preparation requires a structured study schedule that allows for gradual review over an extended period. Moreover, techniques such as active recall and spaced repetition have been shown to significantly improve long-term retention. Adequate rest before an examination is equally paramount to optimal cognitive performance."),

("video game session",
 "Rage quit that boss fight for the third time tonight, the pattern just makes no sense to me. My roommate keeps laughing every time I yell at the screen. Going to try again tomorrow with a clearer head, maybe.",
 "Video gaming has evolved into a multifaceted form of entertainment encompassing narrative depth and complex mechanics. Furthermore, many titles now incorporate sophisticated difficulty curves designed to sustain player engagement over time. This design philosophy underscores the industry's growing emphasis on player retention and satisfaction."),

("laundry day",
 "Left a red sock in with the whites again and now half my shirts have a faint pink tint. Third time this has happened and you'd think I'd learn by now. At least it's an excuse to buy new socks.",
 "Efficient laundry practices can significantly extend the lifespan of clothing while minimizing energy consumption. Moreover, sorting garments by color and fabric type is paramount to preventing dye transfer and damage. Consistent adherence to care labels ultimately preserves the quality of one's wardrobe."),

("first day at new job",
 "Got lost trying to find the bathroom twice and had to awkwardly ask a stranger for directions. Everyone's names are already a blur in my head. At least the free coffee in the break room is a solid perk.",
 "Starting a new job often involves an adjustment period during which employees acclimate to organizational culture and workflow. Furthermore, proactive communication with colleagues can significantly ease this transitional phase. Employers who provide structured onboarding tend to see improved employee retention over the long term."),

("weather complaints",
 "It's been raining for six days straight and my socks are perpetually damp from walking to the car. The dog refuses to go outside for more than thirty seconds now. Ready for literally any sign of sun at this point.",
 "Prolonged periods of adverse weather can significantly impact both individual mood and community infrastructure. Moreover, meteorological patterns influenced by broader climate trends are increasingly difficult to predict with precision. Preparedness and adaptable planning remain essential in mitigating the disruptive effects of extended rainfall."),

("plant care",
 "My fiddle leaf fig keeps dropping leaves no matter what I try and I'm honestly out of ideas at this point. Read online that maybe it's getting too much direct sun by the window. Moved it to the hallway, fingers crossed.",
 "Indoor plant care requires an understanding of species-specific requirements regarding light, humidity, and watering frequency. Furthermore, overwatering remains one of the most common causes of houseplant decline among novice gardeners. Consistent observation and gradual adjustments are paramount to fostering healthy growth."),

("credit card debt",
 "Finally sat down and added up what I actually owe and immediately regretted looking. Going to try the snowball method everyone talks about, starting with the smallest balance first. Not exactly thrilled about it but something has to change.",
 "Managing credit card debt effectively requires a disciplined repayment strategy and a clear understanding of interest accrual. Moreover, prioritizing high-interest balances can substantially reduce the total cost of repayment over time. Financial counseling services can provide valuable guidance for individuals navigating significant debt burdens."),

("morning commute traffic",
 "Sat in the same stretch of highway for almost forty minutes because of some accident up ahead. Missed my usual exit trying to reroute through side streets I barely know. Going to leave twenty minutes earlier tomorrow, probably.",
 "Urban traffic congestion continues to pose significant challenges for commuters and city planners alike. Furthermore, unpredictable incidents such as accidents can drastically extend average travel times during peak hours. Investment in adaptive traffic management systems is paramount to alleviating these recurring bottlenecks."),

("birthday party planning",
 "Ordered the cake three weeks early because the bakery gets slammed that time of year, and I still forgot the balloons until the last minute. Kids showed up an hour before I was even close to ready. Somehow it all came together anyway.",
 "Planning a successful birthday celebration requires careful coordination of logistics, guest lists, and thematic elements. Moreover, early preparation can significantly reduce last-minute stress for the host. A well-organized itinerary ultimately contributes to a memorable experience for both hosts and attendees."),

("solar panels installation",
 "The installer said it'll take about six months before we start actually seeing savings on the electric bill. Watching the little app track our energy production has honestly become a weird obsession of mine. Hoping it pays off before we sell the house someday.",
 "The adoption of residential solar panel systems has grown substantially amid rising energy costs and environmental awareness. Furthermore, long-term savings on utility expenses often offset the initial installation investment within several years. Government incentives continue to play a pivotal role in accelerating this renewable energy transition."),

("pet vet visit",
 "The cat yowled the entire car ride like we were driving her to certain doom, per usual. Turns out it was just a minor ear infection, nothing too serious. Still cost more than I expected for a five minute appointment.",
 "Routine veterinary visits are paramount to the early detection and prevention of potential health issues in pets. Moreover, maintaining a consistent vaccination and check-up schedule can substantially extend an animal's overall lifespan. Pet owners are encouraged to establish a trusted relationship with a local veterinary practice."),

("power outage at home",
 "The power went out right in the middle of dinner and we ended up finishing by candlelight, which was actually kind of fun. Had to dig through three drawers to find a working flashlight. Came back on just as we were about to give up and go to bed.",
 "Power outages can significantly disrupt daily routines and pose challenges for households reliant on electronic devices. Furthermore, maintaining an emergency kit with flashlights and battery backups is paramount to weathering such disruptions. Utility companies continue to invest in grid resilience to minimize the frequency of these events."),

("company reorganization",
 "Half the office got moved to a different floor and nobody really explained why until the group email went out days later. My old team basically doesn't exist anymore in any recognizable form. Trying to stay positive but it's a lot of unknowns right now.",
 "Corporate reorganizations often necessitate significant adjustments to team structures and reporting lines. Moreover, transparent communication throughout this process is paramount to maintaining employee morale and trust. Organizations that manage change effectively tend to experience a smoother transition and reduced turnover."),

("recycling habits",
 "Realized I've been putting the wrong plastics in the recycling bin for who knows how long after reading the sticker more carefully. Kind of embarrassing given how long I've lived here. Going to actually look up the local guidelines this time.",
 "Adopting responsible recycling habits requires households to understand local waste management guidelines and material classifications. Furthermore, contamination from improperly sorted materials can undermine the efficiency of municipal recycling programs. Public education campaigns remain paramount to improving overall recycling compliance."),

("wedding planning stress",
 "The venue changed the catering menu without telling us and now we're scrambling to find something the vegetarian guests can actually eat. Between the flowers and the seating chart I feel like I need a full-time assistant. Only two more months to figure it all out.",
 "Wedding planning often involves coordinating numerous vendors and logistical details within a constrained timeline. Moreover, unexpected changes from suppliers can necessitate rapid adjustments to previously finalized arrangements. Engaging a professional planner can significantly alleviate the stress associated with these complex undertakings."),

("air travel delays",
 "Sat on the tarmac for two hours because of some vague mechanical issue nobody would explain clearly. Missed my connecting flight and ended up sleeping in an airport chair overnight. Never again booking a layover under ninety minutes.",
 "Air travel delays continue to pose significant challenges for passengers navigating tight connection schedules. Furthermore, mechanical and weather-related disruptions remain among the most common causes of extended tarmac waits. Airlines that proactively communicate updates tend to mitigate passenger frustration during such incidents."),

("standing desk switch",
 "My lower back actually feels better after switching to the standing desk, though my feet ache by mid afternoon now instead. Bought one of those cushioned mats which helped a bit. Still figuring out the right balance between sitting and standing throughout the day.",
 "Transitioning to a standing desk has been associated with reduced sedentary behavior and improved posture among office workers. Moreover, gradual adjustment periods are paramount to preventing discomfort in the legs and lower back. Ergonomic accessories such as anti-fatigue mats can further enhance overall comfort."),

("neighborhood dog park",
 "Our dog made a new best friend at the park today, some giant goofy lab that just wanted to chase tennis balls forever. Ended up chatting with the owner for like half an hour without even meaning to. Small talk turned into an actual conversation, which was nice for once.",
 "Dog parks serve a pivotal role in promoting canine socialization and physical exercise within urban communities. Furthermore, these shared spaces often foster incidental social interactions among pet owners. Well-maintained facilities continue to enhance the overall quality of life for both pets and residents."),

("online course progress",
 "Fell behind on the course modules again this week because work got busy, classic story. Finally caught up over the weekend by binge watching three lectures back to back. The quizzes are honestly harder than I expected for an intro level class.",
 "Enrolling in an online course requires disciplined time management to keep pace with structured module deadlines. Moreover, active engagement with quizzes and assignments is paramount to reinforcing conceptual understanding. Self-paced learning platforms continue to underscore the importance of consistent, intentional study habits."),

("farmers market trip",
 "Grabbed way too many peaches because the vendor let me try one and it was ridiculously good. Ended up chatting with the guy selling honey about his bees for longer than I meant to. Came home with basically no plan for half of what I bought.",
 "Farmers markets play a pivotal role in supporting local agriculture and fostering direct relationships between producers and consumers. Furthermore, purchasing seasonal produce often ensures superior freshness and nutritional value. These community-oriented marketplaces continue to underscore the growing demand for sustainable food systems."),

# --- Additional pairs covering blind spots: (a) formal/data-heavy human
# academic or professional writing that should still read as human, and
# (b) AI-style text that avoids obvious "furthermore/paramount" buzzwords
# but still has the flatter, listy, hedge-heavy structure typical of LLMs.
("sleep study results",
 "We recruited 42 participants for six weeks and tracked sleep with both surveys and wrist actigraphy. Screen time before bed was weakly-to-moderately correlated with shorter total sleep, though honestly the effect was smaller than I expected going in, and a couple of outliers in the data made me want to rerun the analysis excluding them.",
 "Here is a summary of what the research shows about sleep. First, screen exposure before bed is linked to shorter sleep duration in most studies. Second, the effect size varies quite a bit depending on age group. Third, more controlled trials are needed before strong conclusions can be drawn about causation versus correlation."),

("team status email",
 "Quick update: the client moved the deadline to next Friday, so if everyone could send me revised timelines by tomorrow that would help a ton. Also does anyone know if Sarah's back from leave, I need her to sign off on the budget before we can lock anything in.",
 "Here is a quick project update for the team. The client has requested that the deadline be moved to next Friday. Please send updated timelines by end of day tomorrow so the schedule can be revised accordingly. Additionally, budget sign-off is still pending and will need to be confirmed before the next phase begins."),

("tips for better sleep",
 "Honestly the thing that helped me most was just keeping the room cold, like uncomfortably cold at first. Also ditching my phone an hour before bed felt impossible for like a week and then suddenly wasn't. Waking up at the same time even on weekends sounds annoying but it genuinely made the biggest difference for me.",
 "Here are three tips for better sleep. First, keep your bedroom cool, ideally between 60 and 67 degrees. Second, avoid screens for at least 30 minutes before bed, since blue light can disrupt melatonin production. Third, try to wake up at the same time every day, even on weekends, to help regulate your internal clock."),

("social media and relationships",
 "I think social media has messed with how my friend group actually talks to each other, not in some huge dramatic way, just little things, like people liking a post instead of actually texting back. It's convenient but sometimes I catch myself scrolling instead of calling someone I haven't talked to in months.",
 "Social media platforms have reshaped how people form and maintain relationships today. While these tools offer unprecedented connectivity, they also introduce challenges around privacy, mental health, and the authenticity of online interactions. Understanding both sides of this shift is important for using these platforms in a healthy way."),

("quarterly sales report",
 "Q3 numbers came in about 4 percent under forecast, mostly because the west region had that warehouse delay in August. Not thrilled about it but the pipeline for Q4 actually looks stronger than last year, so I'm not panicking yet. Going to flag the warehouse issue in tomorrow's meeting.",
 "Here is an overview of the quarterly sales performance. Revenue for Q3 came in approximately 4 percent below forecast, primarily attributed to a warehouse delay in the west region during August. Looking ahead, the Q4 pipeline shows stronger indicators than the prior year. It is recommended that the warehouse delay be addressed in the upcoming review."),

("recipe instructions casual",
 "Just throw the chicken in the marinade for like an hour if you have time, longer if you don't mind planning ahead. Sear it hot and fast so you actually get color on it instead of just steaming the poor thing. Let it rest a few minutes before cutting or all the juice ends up on the cutting board instead of in the meat.",
 "Here is a simple method for cooking marinated chicken. First, marinate the chicken for at least one hour to allow the flavors to develop. Next, sear the chicken over high heat to achieve a golden crust. Finally, allow the chicken to rest for several minutes before slicing to retain its juices."),

("benefits of walking",
 "Started walking to work instead of driving a couple months ago and my knees actually feel better, which surprised me. It's only twenty five minutes but it's become this weird little buffer between home and work that I didn't know I needed. Rain kind of ruins the whole plan though.",
 "Here are some benefits of walking regularly. First, walking can improve cardiovascular health with relatively low impact on joints. Second, it provides a mental transition between different parts of the day, which can reduce stress. Third, consistency matters more than intensity when it comes to seeing long-term benefits."),

("customer feedback analysis",
 "Went through last month's support tickets and the biggest complaint by far, like way more than I expected, was the checkout page timing out on mobile. Only a handful of people mentioned pricing, which actually contradicts what the sales team keeps assuming. Going to bring this up before we plan next quarter's roadmap.",
 "Here is an analysis of recent customer feedback. The most frequent complaint involved the checkout page timing out on mobile devices. Comparatively few users cited pricing as a concern, which contrasts with assumptions held by the sales team. These findings should be considered when planning the next product roadmap."),

# --- More formal-register pairs: human academic/business writing is often
# mistaken for AI purely because it's formal. These pairs teach the model
# that formality alone isn't the signal - genuine human writing still has
# hedging, first-person asides, and slightly uneven structure even when
# it's professional, whereas AI writing is formal *and* uniformly smooth.
("literature review section",
 "Prior work on this topic is honestly a bit scattered. Smith (2019) found a moderate effect, but Chen and Ortiz (2021) couldn't replicate it with a larger sample, which makes me cautious about over-interpreting our own results here. I've tried to flag where the evidence is shakier throughout this section rather than presenting it as settled.",
 "The existing literature on this topic presents a range of findings. Smith (2019) identified a moderate effect, while subsequent work by Chen and Ortiz (2021) reported inconsistent results using a larger sample. This body of research underscores the need for further investigation to establish more definitive conclusions."),

("performance review draft",
 "I want to be upfront that I think Alex has grown a lot this quarter, especially on the client-facing side, but the missed deadlines in March are still something we need to address directly. Not trying to sugarcoat it, just laying out both sides before our one on one.",
 "This performance review highlights significant growth demonstrated by the employee over the past quarter, particularly in client-facing responsibilities. However, it is important to note that missed deadlines in March represent an area requiring further attention. A balanced discussion of these points is recommended during the upcoming one-on-one meeting."),

("grant proposal excerpt",
 "We're asking for funding to run a pilot with about 30 households over four months, which admittedly is a small sample, but it should be enough to tell us if the intervention is worth scaling. Budget details are in the appendix, and I've tried to be conservative rather than padding numbers.",
 "This proposal requests funding to conduct a pilot study involving approximately 30 households over a four-month period. While the sample size is modest, it is expected to provide sufficient preliminary evidence regarding the intervention's scalability. Detailed budget allocations are included in the appendix for review."),

("technical incident postmortem",
 "Root cause ended up being a cache invalidation bug we introduced two deploys ago, not the database migration everyone initially suspected. Took us longer than it should have to catch this because the error logs were misleading. Adding better alerting around cache TTLs so this doesn't slip through again.",
 "The root cause of this incident was identified as a cache invalidation issue introduced in a recent deployment, rather than the database migration initially suspected. Misleading error logs contributed to a delayed diagnosis. Improved alerting around cache configuration is recommended to prevent recurrence of similar incidents."),

("cover letter excerpt",
 "I know my background is more in operations than marketing, but honestly that's part of why I think I'd bring something different to this role. I've spent the last three years fixing broken processes, and a lot of that translates directly to campaign logistics, even if my resume doesn't scream marketing at first glance.",
 "While my professional background is primarily in operations rather than marketing, I believe this experience offers a unique and valuable perspective for this role. Over the past three years, I have focused extensively on optimizing operational processes, skills that translate effectively to campaign logistics and execution. I am confident this diverse background would be an asset to your team."),

("policy memo",
 "Legal flagged a couple of concerns with the new data retention policy, mainly around how long we're keeping support ticket logs. I'd recommend we shorten that window to 12 months instead of 24, though I know that means more storage churn on our end. Open to pushback on this if ops sees a problem.",
 "Legal counsel has identified several concerns regarding the proposed data retention policy, particularly the duration for which support ticket logs are retained. It is recommended that this retention period be reduced from 24 months to 12 months. Feedback from the operations team is welcomed prior to finalizing this policy change."),

# --- Casual-register AI pairs: real AI assistants write casual, upbeat,
# contraction-heavy copy constantly (product blurbs, app descriptions,
# social captions, review summaries). The model was over-relying on
# formal connector words ("furthermore", "moreover") as its main AI
# signal, so casual AI copy slipped through as "Human". These pairs teach
# subtler AI tells that persist even in casual tone: generic praise
# without specific personal detail, safe uniform enthusiasm, and
# formulaic list structure, versus a genuine human's specific, uneven,
# sometimes contradictory personal take.
("app recommendation",
 "Okay so I've tried like five to-do apps this year and this is the first one I haven't deleted after a week. It's not fancy, the sync is a little laggy on wifi honestly, but something about the layout just clicked for my brain finally.",
 "This app is easily one of the best productivity tools available right now. It's simple, intuitive, and packed with features that help you stay organized every day. Whether you're managing work tasks or personal goals, it's a fantastic choice that's sure to boost your efficiency."),

("restaurant review",
 "We went on a Tuesday so it was dead empty, which honestly made the vibe kind of weird for a date night. Food was good though, the garlic bread specifically was unreal, but our server disappeared for like 25 minutes at one point.",
 "This restaurant is a must-visit for anyone who loves great food and a welcoming atmosphere. From the moment you walk in, it's clear that quality and service are top priorities here. The menu offers something for everyone, and it's definitely a spot worth adding to your list."),

("movie opinion",
 "Honestly the pacing in the middle third dragged so much I checked my phone twice, but that ending genuinely got me, not gonna lie. Would probably recommend it but tell people to push through act two.",
 "This movie is an absolute must-watch for fans of the genre. It's got a compelling story, strong performances, and visuals that are simply stunning throughout. Whether you're in it for the action or the emotional depth, it's a film that delivers on every level."),

("skincare product review",
 "Been using this for like three weeks and my skin isn't glowing or anything dramatic, but the redness on my cheeks did calm down a bit, which is honestly all I wanted. Smells kind of medicinal though, not gonna lie.",
 "This skincare product is a game-changer for anyone looking to improve their skin health. It's lightweight, absorbs quickly, and leaves your skin feeling refreshed and hydrated. With consistent use, it's a fantastic addition to any daily skincare routine."),

("fitness tracker review",
 "The step count seems fine but the sleep tracking is wildly off, it thought I was awake during a solid two hours I know I was out cold. Battery life is genuinely great though, I'll give it that.",
 "This fitness tracker is perfect for anyone serious about reaching their health goals. It's packed with accurate tracking features, a sleek design, and a battery that lasts for days. Whether you're a beginner or a seasoned athlete, it's a reliable companion for your fitness journey."),

("travel destination blurb",
 "We almost skipped this town honestly, it was just a random stop because the next hotel was booked up, but the little market by the harbor ended up being the best part of the whole trip.",
 "This destination is a hidden gem that's perfect for travelers seeking adventure and relaxation. With stunning scenery, rich culture, and warm hospitality, it's a destination that's sure to exceed expectations. Whether you're looking for outdoor activities or a peaceful getaway, it's got something for everyone."),

("book blurb casual",
 "Took me forever to get into this one, the first hundred pages are rough, but once the second narrator shows up it completely turned around for me. Ending felt a little rushed though.",
 "This book is a captivating read from start to finish. It's beautifully written, with rich characters and a plot that keeps you hooked until the very last page. Whether you're a casual reader or a dedicated bookworm, it's a story that's sure to leave a lasting impression."),

("headphones review",
 "Bass is honestly a bit much out of the box until you mess with the eq settings in the app, then they sound pretty solid for the price. Case feels kind of cheap though, cracked a corner within a month.",
 "These headphones are an excellent choice for anyone who values great sound quality and comfort. They're lightweight, deliver crisp audio, and offer impressive battery life for all-day use. Whether you're commuting or working out, they're a reliable and versatile option."),

("coffee shop caption",
 "This place has become my whole personality apparently, third time this week. The oat milk latte's good but I mostly come for the corner table by the window, nobody bothers you there.",
 "This coffee shop is the perfect spot to relax, work, or catch up with friends. With its cozy atmosphere, delicious drinks, and friendly staff, it's quickly become a local favorite. Whether you need a quiet corner or a lively hangout spot, it's got you covered."),

("online course review",
 "The first few modules felt kind of slow and repetitive honestly, but it picks up once you get to the project-based section. Wish there was more direct feedback from an actual instructor though.",
 "This online course is an excellent resource for anyone looking to build new skills. It's well-structured, easy to follow, and packed with practical examples that reinforce key concepts. Whether you're a beginner or looking to advance your knowledge, it's a valuable investment in your learning journey."),
]

def build_rows():
    rows = []
    for topic, human, ai in PAIRS:
        rows.append({"text": human, "generated": 0})
        rows.append({"text": ai, "generated": 1})

    # Augment: concatenate two or three same-class paragraphs from *different*
    # topics to create longer, more varied documents (helps the model see
    # AI/human style persist across topic boundaries, and gives TF-IDF more
    # n-gram diversity than the base set alone). Random sampling (fixed seed)
    # instead of a small fixed offset window gives far more unique
    # combinations from the same template pool.
    n = len(PAIRS)
    rng = random.Random(42)
    target_per_class = 2200

    human_paragraphs = [p[1] for p in PAIRS]
    ai_paragraphs = [p[2] for p in PAIRS]

    for _ in range(target_per_class):
        k = rng.choice([2, 2, 3])  # mostly pairs, sometimes triples
        h_pick = rng.sample(human_paragraphs, k=k)
        a_pick = rng.sample(ai_paragraphs, k=k)
        rows.append({"text": " ".join(h_pick), "generated": 0})
        rows.append({"text": " ".join(a_pick), "generated": 1})

    df = pd.DataFrame(rows).drop_duplicates(subset=["text"])
    df["source"] = "synthetic"
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df

if __name__ == "__main__":
    df = build_rows()
    out_dir = os.path.join(os.path.dirname(__file__), "..", "dataset")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "ai_vs_human.csv")
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")
    print(df["generated"].value_counts())
