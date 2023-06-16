// Initialize Firebase (ADD YOUR OWN DATA)
// import { initializeApp } from "firebase/app";
// import { getAnalytics } from "firebase/analytics";

const config = {
    apiKey: "AIzaSyCS6MzsAo9Aq6QRvRgVJ1rYyiDyCGAEsL4",
    authDomain: "g-contact-predict-bay.firebaseapp.com",
    databaseURL: "https://g-contact-predict-bay-default-rtdb.firebaseio.com",
    projectId: "g-contact-predict-bay",
    storageBucket: "g-contact-predict-bay.appspot.com",
    messagingSenderId: "356036840004",
    appId: "1:356036840004:web:a0bb31f5adea5de0e7ca11",
    measurementId: "G-707Y2MG55W"
  };

  firebase.initializeApp(config);
  
//   const app = initializeApp(firebaseConfig);
//   const analytics = getAnalytics(app);
  // Reference messages collection
  var messagesRef = firebase.database().ref('messages');
  
  document.getElementById('contactForm').addEventListener('submit', submitForm);
  
  function submitForm(e){
    e.preventDefault();

    var name = getInputVal('name');
    var company = getInputVal('company');
    var email = getInputVal('email');
    var phone = getInputVal('phone');
    var message = getInputVal('message');
  
    saveMessage(name, company, email, phone, message);
  
    document.querySelector('.alert').style.display = 'block';
  
    setTimeout(function(){
      document.querySelector('.alert').style.display = 'none';
    },3000);
  
    document.getElementById('contactForm').reset();
  }
  
  function getInputVal(id){
    return document.getElementById(id).value;
  }
  
  function saveMessage(name, company, email, phone, message){
    var newMessageRef = messagesRef.push();
    newMessageRef.set({
      name: name,
      company:company,
      email:email,
      phone:phone,
      message:message
    });
  }