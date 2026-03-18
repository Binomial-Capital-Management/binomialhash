plugins {
    `java-library`
    `maven-publish`
    signing
}

group = "com.binomialcapital"
version = "0.1.2"

java {
    sourceCompatibility = JavaVersion.VERSION_17
    targetCompatibility = JavaVersion.VERSION_17
    withJavadocJar()
    withSourcesJar()
}

repositories {
    mavenCentral()
}

dependencies {
    implementation("com.fasterxml.jackson.core:jackson-databind:2.17.0")
    implementation("org.apache.commons:commons-math3:3.6.1")

    compileOnly("org.apache.poi:poi-ooxml:5.2.5")
    compileOnly("com.knuddels:jtokkit:1.1.0")

    testImplementation("org.junit.jupiter:junit-jupiter:5.10.2")
    testImplementation("org.assertj:assertj-core:3.25.3")
    testImplementation("org.apache.poi:poi-ooxml:5.2.5")
    testImplementation("com.knuddels:jtokkit:1.1.0")
}

tasks.test {
    useJUnitPlatform()
}

publishing {
    publications {
        create<MavenPublication>("mavenJava") {
            from(components["java"])
            pom {
                name.set("BinomialHash")
                description.set("Structured payload compaction and manifold-aware analytics for LLM tool outputs")
                url.set("https://github.com/Binomial-Capital-Management/binomialhash-java")
                licenses {
                    license {
                        name.set("MIT License")
                        url.set("https://opensource.org/licenses/MIT")
                    }
                }
                developers {
                    developer {
                        id.set("bcm")
                        name.set("Binomial Capital Management")
                        email.set("support@bcmassets.com")
                    }
                }
                scm {
                    connection.set("scm:git:git://github.com/Binomial-Capital-Management/binomialhash-java.git")
                    developerConnection.set("scm:git:ssh://github.com/Binomial-Capital-Management/binomialhash-java.git")
                    url.set("https://github.com/Binomial-Capital-Management/binomialhash-java")
                }
            }
        }
    }
    repositories {
        maven {
            name = "OSSRH"
            url = uri("https://s01.oss.sonatype.org/service/local/staging/deploy/maven2/")
            credentials {
                username = findProperty("ossrhUsername") as String? ?: System.getenv("OSSRH_USERNAME") ?: ""
                password = findProperty("ossrhPassword") as String? ?: System.getenv("OSSRH_PASSWORD") ?: ""
            }
        }
    }
}

signing {
    sign(publishing.publications["mavenJava"])
}

tasks.javadoc {
    if (JavaVersion.current().isJava9Compatible) {
        (options as StandardJavadocDocletOptions).addBooleanOption("html5", true)
    }
}
