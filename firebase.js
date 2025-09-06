import { initializeApp } from "firebase/app";
import { getFirestore, collection, addDoc, serverTimestamp } from "firebase/firestore";

const firebaseConfig = {
  apiKey: "AIzaSyALVkB5jfl6O0CLNBtGmaX87Kc6UBu2TLE",
  authDomain: "safai-saathi.firebaseapp.com",
  projectId: "safai-saathi",
  storageBucket: "safai-saathi.firebasestorage.app",
  messagingSenderId: "6015045092",
  appId: "1:6015045092:web:a31cf2a86330fac60d4bf1",
  measurementId: "G-ZMM65PNXCL"
};

const app = initializeApp(firebaseConfig);
export const db = getFirestore(app);

export async function saveModelResult(result) {
  try {
    const docRef = await addDoc(collection(db, "model_results"), {
      ...result,
      createdAt: serverTimestamp()
    });
    return docRef.id;
  } catch (error) {
    console.error("Error saving model result:", error);
    throw error;
  }
}
