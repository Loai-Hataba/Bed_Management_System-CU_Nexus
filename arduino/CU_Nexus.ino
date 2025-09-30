
#include <SPI.h>
#include <MFRC522.h>

#define SS_PIN 10
#define RST_PIN 9

MFRC522 mfrc522(SS_PIN, RST_PIN);

// store seen UIDs (max 20 cards for this example)
String seenUIDs[20];
int seenCount = 0;

// Function to check if UID exists in the list
int findUID(String uid) {
  for (int i = 0; i < seenCount; i++) {
    if (seenUIDs[i] == uid) {
      return i; // return index if found
    }
  }
  return -1; // not found
}

// Remove UID from list (when leaving)
void removeUID(int index) {
  for (int i = index; i < seenCount - 1; i++) {
    seenUIDs[i] = seenUIDs[i + 1];
  }
  seenCount--;
}

void setup() {
  Serial.begin(115200);
  SPI.begin();
  mfrc522.PCD_Init();
  Serial.println("RFID Reader Ready...");
}

void loop() {
  // Check if new card is present
  if (!mfrc522.PICC_IsNewCardPresent() || !mfrc522.PICC_ReadCardSerial())
    return;

  // Build UID string
  String uid = "";
  for (byte i = 0; i < mfrc522.uid.size; i++) {
    uid += (mfrc522.uid.uidByte[i] < 0x10 ? "0" : "");
    uid += String(mfrc522.uid.uidByte[i], HEX);
  }
  uid.toUpperCase();

  // Check if UID is in the list
  int index = findUID(uid);

  if (index == -1) {
    // UID not in list → new entry
    if (seenCount < 20) {
      seenUIDs[seenCount] = uid;
      seenCount++;
    }
    Serial.print("UID:"); Serial.print(uid);
    Serial.println(" ON_BED:1");
  } else {
    // UID already in list → remove it
    removeUID(index);
    Serial.print("UID:"); Serial.print(uid);
    Serial.println(" ON_BED:0");
  }

  // Halt PICC
  mfrc522.PICC_HaltA();
}



// #include <SPI.h>
// #include <MFRC522.h>

// #define SS_PIN 10   // SDA pin of RC522
// #define RST_PIN 9   // RST pin of RC522

// MFRC522 mfrc522(SS_PIN, RST_PIN); // Create MFRC522 instance

// // 🔹 Helper: Convert UID to string like "3A:5F:C2:17"
// String getUID(MFRC522::Uid uid) {
//   String uidString = "";
//   for (byte i = 0; i < uid.size; i++) {
//     if (uid.uidByte[i] < 0x10) uidString += "0"; // leading zero
//     uidString += String(uid.uidByte[i], HEX);
//     if (i < uid.size - 1) uidString += ":";
//   }
//   uidString.toUpperCase();
//   return uidString;
// }

// // 🔹 List of authorized cards
// String authorizedCards[] = {
//   "DA:B1:C5:B2"   // card 1
// };
// const int numCards = sizeof(authorizedCards) / sizeof(authorizedCards[0]);

// void setup() {
//   Serial.begin(115200);
//   SPI.begin();
//   mfrc522.PCD_Init();
//   delay(2000);
//   Serial.println("Place your card near the reader...\n");
// }

// void loop() {
//   // Look for new cards
//   if (!mfrc522.PICC_IsNewCardPresent()) return;
//   if (!mfrc522.PICC_ReadCardSerial()) return;

//   // Convert UID to string
//   String uidStr = getUID(mfrc522.uid);
//   Serial.println("Card UID: " + uidStr);

//   // Check if UID is in authorized list
//   bool authorized = false;
//   for (int i = 0; i < numCards; i++) {
//     if (uidStr == authorizedCards[i]) {
//       authorized = true;
//       break;
//     }
//   }

//   if (authorized) {
//     Serial.println("✅ Access Granted!");
//   } else {
//     Serial.println("❌ Access Denied!");
//   }

//   // Halt PICC
//   mfrc522.PICC_HaltA();
//   mfrc522.PCD_StopCrypto1();
// }

