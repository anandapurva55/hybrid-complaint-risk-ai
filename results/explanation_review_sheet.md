# Explanation review sheet (blind A/B)

For each case, read the complaint and the decision, then rate Explanation A and B from 1 (poor) to 5 (excellent) on
**faithful** (says nothing the complaint does not support), **useful** (would help a reviewer act) and **clear**, and mark
which you would rather see. The key to which is the LLM is in `explanation_review_key.json`: do not open it until you finish.

## Case 1
**Complaint:** First of all, mine arrived in a regular denim shade, not the dark wash shown. also, the style is more like a blazer, and it was huge. i am 5'2" and ordered the petite xs but even xxs would have looked like a baggy blazer. as well, the sleeves came barely past my wrists, they are designed to cuff as the model has done? perhaps on someone taller and younger it would look edgy and chic! the quality is excellent with some nice details. this might be one to try in person, if you can.

**Decision:** medium / manual_review (review reasons: none)

**Explanation A**  
The customer received a denim jacket in a different color and style than advertised, and it did not fit as expected. They noted the quality was good but suggested trying it in person.  
The model rated the risk level as medium because the customer's dissatisfaction with the product's appearance and fit is significant, but the quality was still praised.  
Notes: 

**Explanation B**  
Rated medium dissatisfaction risk (high 34%, medium 53%, low 13%). Suggested action: send to a person for review.  
This is a medium-tier case because the review looks mixed or neutral.  
Notes: No review trigger fired.

Ratings (faithful / useful / clear), A: _ / _ / _   B: _ / _ / _   Preferred: A / B / tie

## Case 2
**Complaint:** This jacket is just too small for me anyway and i am no way big. i am tall and athletic and this thing looked like it had shrunk in the wash. there are so many better fitting other types out there, this went back

**Decision:** high / escalate_immediately (review reasons: none)

**Explanation A**  
The customer is dissatisfied with the jacket's size and fit, claiming it appears to have shrunk and does not fit well despite being tall and athletic.  
The model rated the risk level as high due to the customer's dissatisfaction with the product's fit and size, which could indicate a potential issue with product quality or sizing accuracy.  
Notes: 

**Explanation B**  
Rated high dissatisfaction risk (high 71%, medium 26%, low 3%). Suggested action: escalate to a senior agent.  
This is a high-tier case because the models put most of the probability on a strongly unhappy review.  
Notes: No review trigger fired.

Ratings (faithful / useful / clear), A: _ / _ / _   B: _ / _ / _   Preferred: A / B / tie

## Case 3
**Complaint:** Material is so thin this shirt ripped in three places after wearing only once.

got a size larger than i needed as i have a larger chest and even though it's a bit lose the thread has ripped material on my chest and in the armpits.

**Decision:** high / escalate_immediately (review reasons: none)

**Explanation A**  
The customer reported that the shirt material was extremely thin and ripped in multiple places after just one wear, despite ordering a larger size.  
The model rated the risk level as high due to the extremely low probability of low (0.011) and medium (0.044) risk levels, with a high probability of 0.945.  
Notes: 

**Explanation B**  
Rated high dissatisfaction risk (high 94%, medium 4%, low 1%). Suggested action: escalate to a senior agent.  
This is a high-tier case because the models put most of the probability on a strongly unhappy review.  
Notes: No review trigger fired.

Ratings (faithful / useful / clear), A: _ / _ / _   B: _ / _ / _   Preferred: A / B / tie

## Case 4
**Complaint:** I wanted to love this shirt and thought i would based on the reviews. it runs really large however and while i love how flows it is i also looked pregnant. i am keeping though- looks nice layered under a jacket or sweater but alone i feel it looked sloppy and big....... which is sometimes just what you want so it is staying in my closet.

**Decision:** medium / manual_review (review reasons: none)

**Explanation A**  
The customer wanted to love the shirt but found it runs large and looked sloppy when worn alone. They are keeping it for layering.  
The model rated the risk level as medium because the probabilities indicated a moderate risk, with a medium probability of 0.559.  
Notes: 

**Explanation B**  
Rated medium dissatisfaction risk (high 33%, medium 56%, low 11%). Suggested action: send to a person for review.  
This is a medium-tier case because the review looks mixed or neutral.  
Notes: No review trigger fired.

Ratings (faithful / useful / clear), A: _ / _ / _   B: _ / _ / _   Preferred: A / B / tie

## Case 5
**Complaint:** This dress is beautiful. very vibrant and rich looking...however, after wearing it to work the back of the dress was completely stretched out from sitting in my office chair. when i stood up a little bubble remained around my butt. not attractive. i wanted to love it but it's going back.

**Decision:** high / escalate_immediately (review reasons: none)

**Explanation A**  
The customer loved the dress initially but found it stretched out after wearing it to work, leading to an unattractive appearance.  
The model rated the risk level as high due to the high probability (54.6%) of quality issues leading to customer dissatisfaction.  
Notes: 

**Explanation B**  
Rated high dissatisfaction risk (high 55%, medium 40%, low 6%). Suggested action: escalate to a senior agent.  
This is a high-tier case because the models put most of the probability on a strongly unhappy review.  
Notes: No review trigger fired.

Ratings (faithful / useful / clear), A: _ / _ / _   B: _ / _ / _   Preferred: A / B / tie

## Case 6
**Complaint:** I really liked these cords. i ordered both colors. they are very lush corduroy material and the color is very rich. since i am short, i wear them as ankle length (i am 5'1"). my only issue is that they run very large. i am normally a 25-26 and the 25 was still a bit large. unfortunately, they are all sold out of size 24 :(

**Decision:** low / auto_reply (review reasons: none)

**Explanation A**  
The customer liked the cords, praising the material and color, but noted that they run large and are sold out in their preferred size.  
The model rated the risk level as low because the probabilities for low risk were significantly higher (77.5%) compared to medium (18.3%) and high (4.2%).  
Notes: 

**Explanation B**  
Rated low dissatisfaction risk (high 4%, medium 18%, low 78%). Suggested action: reply automatically.  
This is a low-tier case because the review looks clearly positive.  
Notes: No review trigger fired.

Ratings (faithful / useful / clear), A: _ / _ / _   B: _ / _ / _   Preferred: A / B / tie

## Case 7
**Complaint:** I looked forward to this dress.  it's a lovely design.  it's a heavy material that doesn't have the most flattering fit.  the green lace has snags along the hemlines which is a disappointment.  i'm returning the dress.

**Decision:** medium / manual_review (review reasons: none)

**Explanation A**  
Rated medium dissatisfaction risk (high 31%, medium 55%, low 14%). Suggested action: send to a person for review.  
This is a medium-tier case because the review looks mixed or neutral.  
Notes: No review trigger fired.

**Explanation B**  
The customer expresses disappointment with the dress's fit and quality, specifically mentioning snags in the lace, and states they are returning it.  
The model rated the risk level as medium because the probabilities for medium risk were 55.4%, indicating a moderate level of dissatisfaction and potential return.  
Notes: 

Ratings (faithful / useful / clear), A: _ / _ / _   B: _ / _ / _   Preferred: A / B / tie

## Case 8
**Complaint:** Love the wine colored cords-- but i tried them and i was disappointed because my normal size was too small. i usually have to size down in pilcro pants but not these. (i'm an 8, and the 30 wasn't going to button!) i love the slight cropped flare! and the color! did i mention the color? but vanity wouldn't allow me to buy them. hopefully i just got a mislabeled pair!

**Decision:** medium / manual_review (review reasons: none)

**Explanation A**  
The customer loves the color and style of the cords but was disappointed that their normal size was too small.  
The model rated the risk level as medium because the customer expressed dissatisfaction with the sizing, which could indicate a potential issue with product consistency.  
Notes: 

**Explanation B**  
Rated medium dissatisfaction risk (high 4%, medium 24%, low 72%). Suggested action: send to a person for review.  
This is a medium-tier case because the review looks mixed or neutral.  
Notes: No review trigger fired.

Ratings (faithful / useful / clear), A: _ / _ / _   B: _ / _ / _   Preferred: A / B / tie

## Case 9
**Complaint:** This dress has beautiful lines. i loved the design on the material, but the fabric looked very cheap & even was already snagged when it arrived. unfortunately, i must return it.

**Decision:** high / escalate_immediately (review reasons: none)

**Explanation A**  
The customer loved the dress design but found the fabric to be cheap and already snagged upon arrival.  
The model rated the risk level as high due to the customer's dissatisfaction with the fabric quality, which is a significant factor in product returns.  
Notes: 

**Explanation B**  
Rated high dissatisfaction risk (high 56%, medium 38%, low 6%). Suggested action: escalate to a senior agent.  
This is a high-tier case because the models put most of the probability on a strongly unhappy review.  
Notes: No review trigger fired.

Ratings (faithful / useful / clear), A: _ / _ / _   B: _ / _ / _   Preferred: A / B / tie

## Case 10
**Complaint:** I bought this sweater the minute i saw it, and waited for over six weeks for it to come off of back order (it was further delayed from even what i was told when i first ordered it). 
i was beyond excited when it finally arrived and wore the sweater the day after i received it... i had a long flight and thought it was the perfect option for keeping comfy on a plane. 
i was so disappointed that by the time i got off my flight four hours later, the sweater had not one, not two, but three differen

**Decision:** high / escalate_immediately (review reasons: none)

**Explanation A**  
Rated high dissatisfaction risk (high 82%, medium 15%, low 3%). Suggested action: escalate to a senior agent.  
This is a high-tier case because the models put most of the probability on a strongly unhappy review.  
Notes: No review trigger fired.

**Explanation B**  
The customer purchased a sweater that was on back order and experienced significant delays. Upon receiving it, they wore it on a long flight but were disappointed with the quality.  
The model rated the risk level as high due to the customer's dissatisfaction with the product quality and the significant delay in delivery.  
Notes: 

Ratings (faithful / useful / clear), A: _ / _ / _   B: _ / _ / _   Preferred: A / B / tie

## Case 11
**Complaint:** I love the cut and fit of this t. it's soft and has just enough flow to be comfortable without being unflattering. the gray floral print is very pretty. it's perfect with jeans or shorts in the summer.

**Decision:** low / auto_reply (review reasons: none)

**Explanation A**  
Rated low dissatisfaction risk (high 0%, medium 1%, low 99%). Suggested action: reply automatically.  
This is a low-tier case because the review looks clearly positive.  
Notes: No review trigger fired.

**Explanation B**  
The customer expresses satisfaction with the cut, fit, softness, and appearance of the t-shirt, mentioning it is perfect for summer wear.  
The model rated the risk level as low because the probabilities for low risk were very high (0.991), indicating a minimal chance of any issues.  
Notes: 

Ratings (faithful / useful / clear), A: _ / _ / _   B: _ / _ / _   Preferred: A / B / tie

## Case 12
**Complaint:** I wanted to like this coat so bad! i loved the bright red color and design, but when i received it, it seemed very cheap and just too large. i returned it.

**Decision:** high / escalate_immediately (review reasons: none)

**Explanation A**  
The customer expressed disappointment with the quality and size of the coat, leading to a return.  
The model rated the risk level as high due to the customer's dissatisfaction with the product quality and size, which resulted in a return.  
Notes: 

**Explanation B**  
Rated high dissatisfaction risk (high 62%, medium 33%, low 4%). Suggested action: escalate to a senior agent.  
This is a high-tier case because the models put most of the probability on a strongly unhappy review.  
Notes: No review trigger fired.

Ratings (faithful / useful / clear), A: _ / _ / _   B: _ / _ / _   Preferred: A / B / tie

## Case 13
**Complaint:** At the beginning of summer, i first ordered one of these tops in size xl. i liked the colors and the material was such a soft, open weave. i thought it would be perfect for the hot weather so i ordered a 2nd on sale. unfortunately, even though these are made of mesh, these tops ended up being really hot. i wore them in texas, new york, new jersey, and florida. i thought they'd be a good, pretty choice. but, the material with the gathered waist & wrists, trapped in the heat. it was just too sheer

**Decision:** medium / manual_review (review reasons: none)

**Explanation A**  
The customer ordered two tops in size XL, liking the colors and material initially, but found them too hot to wear due to the material trapping heat.  
The model rated the risk level as medium because the customer experienced discomfort due to the tops being too hot, but the complaint did not indicate severe dissatisfaction or safety concerns.  
Notes: 

**Explanation B**  
Rated medium dissatisfaction risk (high 27%, medium 55%, low 18%). Suggested action: send to a person for review.  
This is a medium-tier case because the review looks mixed or neutral.  
Notes: No review trigger fired.

Ratings (faithful / useful / clear), A: _ / _ / _   B: _ / _ / _   Preferred: A / B / tie

## Case 14
**Complaint:** These shipped fast!!! the fit is good. these shorts have some stretch to them, which is nice. there is a green/blue color in the fine detail making them versatile with a few different colors of tops. true to size. the length is perfect for my taste. i am 5'7".

**Decision:** low / auto_reply (review reasons: none)

**Explanation A**  
The customer is highly satisfied with the fast shipping, good fit, stretch, color versatility, true-to-size nature, and perfect length of the shorts.  
The model rated the risk level as low because the probabilities for low risk were 98.9%, indicating a very low chance of any issues.  
Notes: 

**Explanation B**  
Rated low dissatisfaction risk (high 0%, medium 1%, low 99%). Suggested action: reply automatically.  
This is a low-tier case because the review looks clearly positive.  
Notes: No review trigger fired.

Ratings (faithful / useful / clear), A: _ / _ / _   B: _ / _ / _   Preferred: A / B / tie

## Case 15
**Complaint:** This shirt is thin and shapeless. the quality is h&m esque. unflattering. not a fan

**Decision:** high / escalate_immediately (review reasons: none)

**Explanation A**  
Rated high dissatisfaction risk (high 92%, medium 6%, low 1%). Suggested action: escalate to a senior agent.  
This is a high-tier case because the models put most of the probability on a strongly unhappy review.  
Notes: No review trigger fired.

**Explanation B**  
The customer expressed dissatisfaction with the shirt's thinness, shapelessness, and overall quality, comparing it unfavorably to H&M products.  
The model rated the risk level as high due to the customer's strong negative feedback about the product's quality and fit, with a high probability of 0.922.  
Notes: 

Ratings (faithful / useful / clear), A: _ / _ / _   B: _ / _ / _   Preferred: A / B / tie

## Case 16
**Complaint:** I like the contrast of the skirt with the bodice, the length of the sleeves, adn the subtle cut out at the neck. the cut is very flattering, adn i am glad the daker color is at the top, hte bottome color would wash me out. i tried on the regular size nad i would not get a peitte in this as the size was the perfect length for me. petite would be too short for my liking. (regular hit just above the knee)

**Decision:** low / auto_reply (review reasons: none)

**Explanation A**  
Rated low dissatisfaction risk (high 0%, medium 3%, low 97%). Suggested action: reply automatically.  
This is a low-tier case because the review looks clearly positive.  
Notes: No review trigger fired.

**Explanation B**  
The customer appreciates the contrast, length, and cut of the skirt, and is satisfied with the size they tried on.  
The model rated the risk level as low due to the positive feedback and satisfaction expressed by the customer, with a final score of 0.018 and a 96.7% probability of being low risk.  
Notes: 

Ratings (faithful / useful / clear), A: _ / _ / _   B: _ / _ / _   Preferred: A / B / tie

## Case 17
**Complaint:** The dress is too big. i got size 4, after checking the measurements chart (for reference: 5'7", 125 lb). it is too wide, not as fitted as in the picture. the cut is too deep (might work better for a bustier lady, but looks pathetic on me). 

the pattern is too bright, much brighter that in the picture (i got red pattern). it's not that the photos represent the colors inaccurately, it's just... the colors are much more vibrant in real life.

the material is heavier than i expected.

**Decision:** high / escalate_immediately (review reasons: none)

**Explanation A**  
Rated high dissatisfaction risk (high 74%, medium 23%, low 3%). Suggested action: escalate to a senior agent.  
This is a high-tier case because the models put most of the probability on a strongly unhappy review.  
Notes: No review trigger fired.

**Explanation B**  
The customer is dissatisfied with the dress size, fit, color vibrancy, and material weight, which do not match the product description and images.  
The model rated the risk level as high due to the significant discrepancies between the product description and the customer's experience, as indicated by the high probability of 0.741.  
Notes: 

Ratings (faithful / useful / clear), A: _ / _ / _   B: _ / _ / _   Preferred: A / B / tie

## Case 18
**Complaint:** I really wanted to like this shirt, but it made me look 9 months pregnant. which would be fine if i were pregnant.  it sits wired in the bust and flows out in the front and back. not flattering at all.

**Decision:** high / escalate_immediately (review reasons: none)

**Explanation A**  
Rated high dissatisfaction risk (high 91%, medium 8%, low 1%). Suggested action: escalate to a senior agent.  
This is a high-tier case because the models put most of the probability on a strongly unhappy review.  
Notes: No review trigger fired.

**Explanation B**  
The customer expresses dissatisfaction with the shirt's fit, stating it is unflattering and makes them look 9 months pregnant.  
The model rated the risk level as high due to the customer's strong dissatisfaction with the product's fit, which is a significant concern for the brand.  
Notes: 

Ratings (faithful / useful / clear), A: _ / _ / _   B: _ / _ / _   Preferred: A / B / tie
