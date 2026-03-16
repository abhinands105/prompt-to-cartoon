// -----------------------------
// NAVIGATION SYSTEM
// -----------------------------

function nav(id){

document.querySelectorAll(".pane").forEach(pane=>{
pane.classList.remove("active")
})

document.getElementById(id).classList.add("active")

}



// -----------------------------
// PROMPT STATE
// -----------------------------

let selectedStyle = ""
let promptTokens = []

function addToken(token){

if(!promptTokens.includes(token)){
promptTokens.push(token)
}

updatePrompt()

}

function updatePrompt(){

document.getElementById("prompt-input").value =
promptTokens.join(", ")

}

function clearPrompt(){

promptTokens=[]
updatePrompt()

}



// -----------------------------
// LORA MODEL DATABASE
// -----------------------------

const loraDB = {

"Adventure Time":{
img:"styles/adventure_time.jpg",
tags:{
style:["advtime_style"],

characters:[
"finn_char","jake_char","marceline_char","iceking_char","pbubblegum_char","bmo_char"
],

environment:[
"candy_kingdom","treehouse_interior","grassy_field","purple_desert","magical_forest"
],

action:[
"heroic_pose","running_pose","jumping_pose","sword_attack","dynamic_action_pose"
],

camera:[
"close_up","wide_shot","cinematic_view","low_angle","high_angle"
]
}
},

"Avatar":{
img:"styles/avatar.jpg",
tags:{
style:["avatar_comic_style"],

characters:[
"aang_char","katara_char","sokka_char","zuko_char","toph_char","azula_char"
],

environment:[
"forest_clearing","mountain_cliff","village_street","desert_landscape","palace_courtyard"
],

action:[
"airbending_attack","waterbending_wave","earthbending_strike","firebending_attack"
],

camera:[
"wide_shot","cinematic_view","dynamic_perspective","low_angle"
]
}
},

"Powerpuff Girls":{
img:"styles/powerpuff.jpg",
tags:{
style:["ppg_comic_style"],

characters:[
"blossom_char","bubbles_char","buttercup_char","mojo_jojo_char"
],

environment:[
"townsville_city","city_street","park_environment","laboratory_room"
],

action:[
"flying_pose","battle_scene","hero_pose","laser_attack"
],

camera:[
"close_up","wide_shot","comic_panel_view"
]
}
},

"Samurai Jack":{
img:"styles/samurai_jack.jpg",
tags:{
style:["samurai_jack_style"],

characters:[
"jack_char","aku_char","scotsman_char"
],

environment:[
"forest_environment","ancient_temple","desert_canyon","mountain_peak"
],

action:[
"katana_swing","combat_pose","battle_scene","running_pose"
],

camera:[
"cinematic_view","dramatic_scene","wide_shot"
]
}
},

"Steven Universe":{
img:"styles/steven_universe.jpg",
tags:{
style:["su_cartoon_style"],

characters:[
"steven_char","garnet_char","pearl_char","amethyst_char"
],

environment:[
"beach_city","city_street","outer_space","crystal_temple"
],

action:[
"group_reaction","running_pose","dramatic_scene","emotional_pose"
],

camera:[
"close_up","wide_shot","cinematic_view"
]
}
},

"Over The Garden Wall":{
img:"styles/garden_wall.jpg",
tags:{
style:["otgw_style"],

characters:[
"wirt_char","greg_char","woodsman_char"
],

environment:[
"autumn_forest","forest_path","village_street","dark_forest"
],

action:[
"walking_pose","standing_pose","exploring_scene"
],

camera:[
"storybook_scene","wide_shot","cinematic_view"
]
}
},

"Dexter Lab":{
img:"styles/dexter.jpg",
tags:{
style:["dexlab_style"],

characters:[
"dexter_char","dee_dee_char","mandark_char"
],

environment:[
"laboratory_interior","science_lab","classroom_interior"
],

action:[
"experiment_scene","running_pose","invention_scene"
],

camera:[
"close_up","wide_shot"
]
}
},

"Bugs Bunny":{
img:"styles/bugs.jpg",
tags:{
style:["lt_comic_style"],

characters:[
"bugs_char","daffy_char","elmer_fudd_char","tweety_char","sylvester_char"
],

environment:[
"forest_background","desert_canyon","farm_setting"
],

action:[
"running_pose","chase_scene","comic_explosion"
],

camera:[
"comic_panel_view","wide_shot"
]
}
},

"Peanuts":{
img:"styles/peanuts.jpg",
tags:{
style:["peanuts_strip_style"],

characters:[
"charlie_brown_char","snoopy_char","lucy_char","linus_char"
],

environment:[
"grassy_field","school_classroom","baseball_field"
],

action:[
"sitting_pose","walking_pose","conversation_scene"
],

camera:[
"comic_panel_view","close_up"
]
}
},

"Garfield":{
img:"styles/garfield.jpg",
tags:{
style:["garfield_comic_style"],

characters:[
"garfield_char","odie_char","jon_char","liz_char"
],

environment:[
"living_room","kitchen_interior","beach_environment"
],

action:[
"running_pose","jumping_pose","lazy_pose"
],

camera:[
"wide_shot","close_up"
]
}
},

"Looney Tunes":{
img:"styles/looney.jpg",
tags:{
style:["lt_comic_style"],

characters:[
"bugs_char","daffy_char","porky_char","taz_char"
],

environment:[
"cartoon_desert","city_street","forest_background"
],

action:[
"comic_chase","explosion_scene","dynamic_action_pose"
],

camera:[
"wide_shot","comic_panel_view"
]
}
},

"Alice Forever After":{
img:"styles/alice.jpg",
tags:{
style:["alicefa_style"],

characters:[
"alice_char","mad_hatter_char","cheshire_cat_char"
],

environment:[
"magical_forest","tea_party_scene","fantasy_meadow"
],

action:[
"walking_pose","dream_scene"
],

camera:[
"storybook_scene","cinematic_view"
]
}
},

"PB Monkey Jelly":{
img:"styles/monkey.jpg",
tags:{
style:["pbmj_style"],

characters:[
"monkey_char","scientist_char","villain_char"
],

environment:[
"jungle_environment","laboratory_environment","desert_road"
],

action:[
"action_pose","running_pose"
],

camera:[
"close_up","wide_shot"
]
}
},

"Donald Marvel":{
img:"styles/donald.jpg",
tags:{
style:["dm_marvel_style"],

characters:[
"donald_char","mickey_char","goofy_char"
],

environment:[
"city_skyline","space_station","battlefield"
],

action:[
"flying_pose","energy_blast","superhero_pose"
],

camera:[
"cinematic_view","low_angle"
]
}
},

"Wrestle Heist":{
img:"styles/wrestle.jpg",
tags:{
style:["wrestle_style"],

characters:[
"wrestler_char","champion_char","fighter_char"
],

environment:[
"wrestling_arena","stadium_ring"
],

action:[
"fight_pose","jump_attack","crowd_scene"
],

camera:[
"dramatic_view","wide_shot"
]
}
},

"Cartoon Classic":{
img:"styles/classic.jpg",
tags:{
style:["classic_cartoon_style"],

characters:[
"cartoon_hero","cartoon_villain","cartoon_animal"
],

environment:[
"city_street","forest_background","space_background"
],

action:[
"hero_pose","battle_scene","dynamic_action_pose"
],

camera:[
"wide_shot","cinematic_view"
]
}
}

}



// -----------------------------
// STYLE SELECTION
// -----------------------------

function selectStyle(style){

selectedStyle = style
promptTokens=[]

const data = loraDB[style]

addToken(data.tags.style[0])

// preview image

const img = document.getElementById("style-preview")

img.classList.add("change")

setTimeout(()=>{

img.src=data.img
img.classList.remove("change")

},200)

document.getElementById("current-style-title").innerText = style

renderButtons(data)

nav("generator")

}



// -----------------------------
// BUTTON RENDERER
// -----------------------------

function renderButtons(data){

const tagBox=document.getElementById("suggestion-tags")

tagBox.innerHTML=""

function createSection(title,tokens){

const section=document.createElement("div")

const header=document.createElement("h4")
header.innerText=title

section.appendChild(header)

tokens.forEach(tag=>{

const btn=document.createElement("button")

btn.className="tag"

btn.innerText=tag
.replace("_char","")
.replaceAll("_"," ")

btn.onclick=()=>addToken(tag)

section.appendChild(btn)

})

tagBox.appendChild(section)

}

createSection("Characters",data.tags.characters)
createSection("Environment",data.tags.environment)
createSection("Action",data.tags.action)

}



// -----------------------------
// RANDOM PROMPT
// -----------------------------

function randomPrompt(){

const prompts=[

"hero exploring magical forest",

"cute cartoon dragon flying",

"robot fighting ninja",

"princess in fantasy castle",

"cartoon scientist building robot"

]

let random=prompts[Math.floor(Math.random()*prompts.length)]

document.getElementById("prompt-input").value=random

}



// -----------------------------
// GENERATE IMAGE
// -----------------------------
async function generateImage(){

let prompt=document.getElementById("prompt-input").value

if(prompt===""){
alert("Please enter a prompt")
return
}

const btn=document.querySelector(".full-width")
btn.innerText="Generating..."

const loader=document.getElementById("loader")
loader.style.display="block"

try{

const response = await fetch(
"https://glowing-winner-x5vx66r9q766c96vj-8014.app.github.dev/generate",
{
method:"POST",
headers:{
"Content-Type":"application/json"
},
body:JSON.stringify({
prompt:prompt,
style:selectedStyle
})
}
)

if(!response.ok){
throw new Error("API error")
}

const blob = await response.blob()

const imgURL = URL.createObjectURL(blob)

const result=document.getElementById("result-image")

result.src = imgURL
result.style.display="block"

}catch(error){

console.error(error)
alert("Generation failed")

}

loader.style.display="none"
btn.innerText="Generate Cartoon"

}

// -----------------------------
// START PAGE
// -----------------------------

nav("home")



// -----------------------------
// CARD ANIMATION
// -----------------------------

window.addEventListener("load",()=>{

const cards=document.querySelectorAll(".style-card")

cards.forEach((card,i)=>{

card.style.opacity="0"
card.style.transform="translateY(40px)"

setTimeout(()=>{

card.style.transition="all .6s"
card.style.opacity="1"
card.style.transform="translateY(0)"

},i*100)

})

})



